"""
Cahn-Hilliard Learning Script
Learns the free energy derivative using neural networks and adjoint methods.
"""

from firedrake import *
from firedrake.adjoint import *
import os

# Set threading limits to avoid conflicts with MPI
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
import torch

import time
import wandb
import matplotlib
matplotlib.use("Agg")

from training_utils import (
    parse_arguments, setup_device, setup_output_dir, initialize_training
)
from models.dfdc import FEDerivative
from simulation import CHSolver, load_target_data
from checkpoint import save_checkpoint
from plotting import plot_loss_vs_epochs

# Limit PyTorch threads
torch.set_num_threads(1)
torch.set_num_interop_threads(1)


def setup_problem(num_timesteps, data_index=1, N_mesh=100, L_domain=1.0):
    """Setup the problem: mesh, function spaces, and target data."""
    # Create 2D mesh and function spaces (always square)
    mesh = RectangleMesh(N_mesh, N_mesh, L_domain, L_domain)
    V = FunctionSpace(mesh, "Lagrange", 1)
    W = V * V * V
    
    # Load target data
    c_target_list, eta_target_list, _ = load_target_data(num_timesteps, V, None, 0, data_index=data_index)
    
    # Setup initial condition
    u_ic = Function(W, name="Initial_condition")
    print("Setting initial condition from the first timestep of the target data.", flush=True)
    u_ic.sub(0).assign(c_target_list[0])
    u_ic.sub(1).assign(0.0)
    u_ic.sub(2).assign(eta_target_list[0])
    
    # Create solution and test functions
    u = Function(W, name="Solution")
    c, mu, eta = split(u)
    v = TestFunction(W)
    c_test, mu_test, eta_test = split(v)
    
    return V, W, u_ic, u, c, mu, eta, c_test, mu_test, eta_test, c_target_list, eta_target_list


def compute_loss_and_gradient(u_curr, target, device, weight=1.0, sub_index=0):
    """Compute FFT-based loss and its gradient (no truncation). Returns (weighted, raw, grad)."""
    u_curr_np = u_curr.sub(sub_index).dat.data_ro
    target_np = target.dat.data_ro
    
    u_tensor = torch.tensor(u_curr_np, device=device, requires_grad=True)
    t_tensor = torch.tensor(target_np, device=device)
    
    fft_u = torch.fft.fftn(u_tensor)
    fft_t = torch.fft.fftn(t_tensor)
    
    raw_loss = 0.5 * torch.mean(torch.abs(fft_u - fft_t)**2)
    weighted_loss = weight * raw_loss
    
    weighted_loss.backward()
    grad_u_tensor = u_tensor.grad
    
    return weighted_loss.item(), raw_loss.item(), grad_u_tensor


def train_epoch(epoch, num_epochs, model, optimizer, device, u_ic, u, c_target_list, eta_target_list, 
                V, W, dt, M, lmbda, num_timesteps, vtk_out, ch_solver,
                c_snapshots, eta_snapshots, dfdc_outputs, dfdeta_outputs, g_c_list, g_eta_list,
                eta_loss_weight=1.0, integrability_weight=1.0, integrability_start_epoch=0, use_wandb=True,
                should_collect_data=False):
    """Execute one training epoch."""
    # Clear previous tape
    get_working_tape().clear_tape()
    
    epoch_t0 = time.perf_counter()
    continue_annotation()
    
    # Reset solution
    u_curr = u_ic.copy(deepcopy=True)
    
    # --- FORWARD PASS ---
    J_adj = 0.0  # Adjoint functional
    
    # Accumulators for logging
    J_total_weighted = 0.0
    J_total_true = 0.0
    J_c_true = 0.0
    J_eta_weighted = 0.0
    J_eta_true = 0.0
    
    comparison_data = []

    for i in range(num_timesteps):
        c_curr = u_curr.sub(0)
        eta_curr = u_curr.sub(2)
        
        # Snapshots for backprop
        c_snapshot = c_snapshots[i]
        c_snapshot.assign(c_curr)
        
        eta_snapshot = eta_snapshots[i]
        eta_snapshot.assign(eta_curr)
        
        # Neural network prediction
        c_vec = c_curr.dat.data_ro.copy().astype(np.float64)
        eta_vec = eta_curr.dat.data_ro.copy().astype(np.float64)
        with torch.no_grad():
            c_tensor = torch.from_numpy(c_vec.reshape(-1, 1)).to(device).to(torch.float64)
            eta_tensor = torch.from_numpy(eta_vec.reshape(-1, 1)).to(device).to(torch.float64)
            input_tensor = torch.cat([c_tensor, eta_tensor], dim=1)
            # Prediction: [num_dofs, 2]
            preds_np = model(input_tensor).cpu().numpy()
            dfdc_np = preds_np[:, 0]
            dfdeta_np = preds_np[:, 1]
        
        # Update existing dfdc and dfdeta Function for solver and adjoint tracking
        dfdc_f = dfdc_outputs[i]
        dfdc_f.dat.data[:] = dfdc_np

        dfdeta_f = dfdeta_outputs[i]
        dfdeta_f.dat.data[:] = dfdeta_np
        
        # Solve one timestep using CHSolver
        u_next = ch_solver.solve_step(u_curr, dfdc_f, dfdeta_f, u)
        u_curr.assign(u_next)
        
        # Store comparison data for visualization (both c and eta)
        if (i + 1) % 20 == 0 or (i + 1) == num_timesteps:
            comparison_data.append((i, 
                                    u_curr.sub(0).copy(deepcopy=True), c_target_list[i],
                                    u_curr.sub(2).copy(deepcopy=True), eta_target_list[i]))
        
        # --- LOSS CALCULATION ---
        # Loss on concentration (c) - weight is always 1.0
        l_c_w, l_c_t, grad_c_tensor = compute_loss_and_gradient(
            u_curr, c_target_list[i], device, sub_index=0)
        
        # Loss on crystallinity (eta)
        l_eta_w, l_eta_t, grad_eta_tensor = compute_loss_and_gradient(
            u_curr, eta_target_list[i], device,
            weight=eta_loss_weight, sub_index=2)
        
        # Inject gradients into Firedrake adjoint
        g_c_i = g_c_list[i]
        g_c_i.dat.data[:] = grad_c_tensor.cpu().numpy()
        
        g_eta_i = g_eta_list[i]
        g_eta_i.dat.data[:] = grad_eta_tensor.cpu().numpy()
        
        J_adj += assemble(inner(g_c_i, u_curr.sub(0)) * dx)
        J_adj += assemble(inner(g_eta_i, u_curr.sub(2)) * dx)
        
        J_total_weighted += l_c_w + l_eta_w
        J_total_true += l_c_t + l_eta_t
        J_c_true += l_c_t
        J_eta_weighted += l_eta_w
        J_eta_true += l_eta_t
        
        # Write VTK output on final epoch
        if epoch == num_epochs - 1:
            t = (i + 1) * dt
            vtk_out.write(project(u_curr.sub(0), V, name="Volume Fraction"), 
                          project(u_curr.sub(2), V, name="Crystallinity"), time=t)
    
    pause_annotation()
    
    # --- ADJOINT GRADIENT ---
    controls = []
    for i in range(num_timesteps):
        controls.append(Control(dfdc_outputs[i]))
        controls.append(Control(dfdeta_outputs[i]))
        
    rf = ReducedFunctional(J_adj, controls)
    grads = rf.derivative()
    
    # --- PYTORCH BACKPROPAGATION ---
    optimizer.zero_grad()
    
    total_scalar_for_backprop = torch.tensor(0.0, dtype=torch.float64, device=device)
    total_loss_int_true = 0.0
    
    for i in range(num_timesteps):
        c_i_vec = c_snapshots[i].dat.data_ro.copy().astype(np.float64)
        eta_i_vec = eta_snapshots[i].dat.data_ro.copy().astype(np.float64)
        c_i_tensor = torch.from_numpy(c_i_vec.reshape(-1, 1)).to(device).to(torch.float64)
        eta_i_tensor = torch.from_numpy(eta_i_vec.reshape(-1, 1)).to(device).to(torch.float64)
        input_i_tensor = torch.cat([c_i_tensor, eta_i_tensor], dim=1)
        
        if epoch >= integrability_start_epoch and integrability_weight > 0:
            input_i_tensor.requires_grad = True
            
        preds_i_tensor = model(input_i_tensor)
        dfdc_i_pred = preds_i_tensor[:, 0]
        dfdeta_i_pred = preds_i_tensor[:, 1]
        
        sens_c_i = torch.from_numpy(grads[2*i].dat.data_ro.copy().astype(np.float64)).to(device).to(torch.float64)
        sens_eta_i = torch.from_numpy(grads[2*i+1].dat.data_ro.copy().astype(np.float64)).to(device).to(torch.float64)
        
        data_loss_term = torch.sum(dfdc_i_pred * sens_c_i) + torch.sum(dfdeta_i_pred * sens_eta_i)
        
        if epoch >= integrability_start_epoch and integrability_weight > 0:
            grad_dfdc = torch.autograd.grad(dfdc_i_pred, input_i_tensor, grad_outputs=torch.ones_like(dfdc_i_pred), create_graph=True, retain_graph=True)[0]
            d_dfdc_d_eta = grad_dfdc[:, 1]
            grad_dfdeta = torch.autograd.grad(dfdeta_i_pred, input_i_tensor, grad_outputs=torch.ones_like(dfdeta_i_pred), create_graph=True, retain_graph=True)[0]
            d_dfdeta_d_c = grad_dfdeta[:, 0]
            
            loss_int_t = torch.mean((d_dfdc_d_eta - d_dfdeta_d_c)**2)
            total_loss_int_true += loss_int_t.item()
            total_scalar_for_backprop = total_scalar_for_backprop + data_loss_term + integrability_weight * loss_int_t
        else:
            total_scalar_for_backprop = total_scalar_for_backprop + data_loss_term

    if total_scalar_for_backprop.requires_grad:
        total_scalar_for_backprop.backward()
    
    optimizer.step()

    loss_int_avg_true = total_loss_int_true / num_timesteps
    loss_int_avg_weighted = (integrability_weight * total_loss_int_true) / num_timesteps if epoch >= integrability_start_epoch else 0.0
    
    # Include integrability in the totals for logging
    J_total_weighted += loss_int_avg_weighted * num_timesteps
    J_total_true += loss_int_avg_true * num_timesteps

    processed_comparison_data = []
    if should_collect_data:
        processed_comparison_data = [
            (i, u_c_pred.dat.data_ro.copy().astype(np.float32), c_targ.dat.data_ro.copy().astype(np.float32),
                u_eta_pred.dat.data_ro.copy().astype(np.float32), eta_targ.dat.data_ro.copy().astype(np.float32))
            for i, u_c_pred, c_targ, u_eta_pred, eta_targ in comparison_data
        ]
    
    elapsed_time = time.perf_counter() - epoch_t0
    
    return (J_total_weighted, J_total_true, J_c_true, J_eta_weighted, J_eta_true, 
            loss_int_avg_weighted, loss_int_avg_true, elapsed_time, u_curr, processed_comparison_data)


def save_npz_data(output_dir, epoch, preds_collection, epochs_collection, 
                  target_final_global, all_epochs_comparison_data, 
                  epoch_losses, epoch_numbers, model, device, use_wandb, all_nn_outputs,
                  chi, chi_ac, N1, N2, Weta, z, z0, T, dt, M, lmbda, lmbda_eta, L_kinetic,
                  mesh_coords=None,
                  epoch_losses_weighted=None,
                  epoch_losses_true=None,
                  epoch_losses_c_true=None,
                  epoch_losses_eta_weighted=None,
                  epoch_losses_eta_true=None,
                  epoch_losses_int_weighted=None,
                  epoch_losses_int_true=None):
    """Save post-processing data to .npz file."""
    print(f"Saving .npz data at epoch {epoch}...", flush=True)
    
    # Create 2D grid for c and eta (50x50 = 2500 points)
    n_points = 50
    c_lin = np.linspace(0, 1, n_points)
    eta_lin = np.linspace(0, 1, n_points)
    C_grid, ETA_grid = np.meshgrid(c_lin, eta_lin)
    
    c_flat = C_grid.flatten().reshape(-1, 1)
    eta_flat = ETA_grid.flatten().reshape(-1, 1)
    
    # Ensure tensors are float64 to match default NN weights (double precision)
    c_tensor_nn = torch.from_numpy(c_flat).to(device).to(torch.float64)
    eta_tensor_nn = torch.from_numpy(eta_flat).to(device).to(torch.float64)
    input_tensor_nn = torch.cat([c_tensor_nn, eta_tensor_nn], dim=1)
    
    model.eval()
    with torch.no_grad():
        nn_output_values = model(input_tensor_nn).cpu().numpy()
    model.train()

    # Reshape output to [n_points, n_points, 2] for 3D plotting
    nn_output_reshaped = nn_output_values.reshape(n_points, n_points, 2)
    all_nn_outputs.append({'epoch': epoch, 'output': nn_output_reshaped})
    
    npz_path = output_dir / "post_processing_data.npz"
    save_dict = dict(
             preds_collection=np.array(preds_collection),
             epochs_collection=np.array(epochs_collection),
             target_final_global=target_final_global,
             all_epochs_comparison_data=np.array(all_epochs_comparison_data, dtype=object),
             c_values_nn=c_lin,
             eta_values_nn=eta_lin,
             all_nn_outputs=np.array(all_nn_outputs, dtype=object),
             epoch_losses_weighted=np.array(epoch_losses_weighted) if epoch_losses_weighted is not None else np.array([]),
             epoch_losses_true=np.array(epoch_losses_true) if epoch_losses_true is not None else np.array([]),
             epoch_losses_c_true=np.array(epoch_losses_c_true) if epoch_losses_c_true is not None else np.array([]),
             epoch_losses_eta_weighted=np.array(epoch_losses_eta_weighted) if epoch_losses_eta_weighted is not None else np.array([]),
             epoch_losses_eta_true=np.array(epoch_losses_eta_true) if epoch_losses_eta_true is not None else np.array([]),
             epoch_losses_int_weighted=np.array(epoch_losses_int_weighted) if epoch_losses_int_weighted is not None else np.array([]),
             epoch_losses_int_true=np.array(epoch_losses_int_true) if epoch_losses_int_true is not None else np.array([]),
             epoch_numbers=np.array(epoch_numbers),
             nn_output_label=np.array("df/dc and df/deta"),
             chi=np.array(chi),
             chi_ac=np.array(chi_ac),
             N1=np.array(N1),
             N2=np.array(N2),
             Weta=np.array(Weta),
             z=np.array(z),
             z0=np.array(z0),
             T=np.array(T),
             dt=np.array(dt),
             M=np.array(M),
             lmbda=np.array(lmbda),
             lmbda_eta=np.array(lmbda_eta),
             L=np.array(L_kinetic))
    
    # Save 2D mesh node coordinates for post-processing
    if mesh_coords is not None:
        save_dict['mesh_coords'] = mesh_coords
    
    np.savez(npz_path, **save_dict)
    
    if use_wandb:
        wandb.save(str(npz_path))


def main():
    args = parse_arguments()
    
    # Override epochs if profiling
    if args.profile:
        print("Profiling mode enabled: reducing epochs to 2", flush=True)
        args.epochs = 2
    
    output_dir = setup_output_dir(args)
    device = setup_device(args)
    
    # Problem parameters
    dt = args.dt
    num_timesteps = 100
    T = num_timesteps * dt
    M = args.M
    lmbda = 5e-2
    
    # Setup problem (2D RectangleMesh)
    V, W, u_ic, u, c, mu, eta, c_test, mu_test, eta_test, c_target_list, eta_target_list = setup_problem(
        num_timesteps, data_index=args.data_index, N_mesh=args.N, L_domain=args.L
    )
    
    # Get mesh coordinates for post-processing (shape: (n_dofs, 2))
    mesh_coords = V.mesh().coordinates.dat.data_ro.copy()
    
    # Initialize solver
    ch_solver = CHSolver(W, dt, M, lmbda, lmbda_eta=args.lmbda_eta, L=args.L_kinetic)
    
    # Create model
    model = FEDerivative()
    
    # Initialize training
    model, optimizer, scheduler, start_epoch, epoch_losses, epoch_numbers = initialize_training(
        args, model, device, output_dir, checkpoint_filename="ch_learn_dfdc.pth"
    )
    
    # Initialize component losses
    epoch_losses_weighted = []
    epoch_losses_true = []
    epoch_losses_c_true = []
    epoch_losses_eta_weighted = []
    epoch_losses_eta_true = []
    epoch_losses_int_weighted = []
    epoch_losses_int_true = []
    
    num_epochs = args.epochs
    checkpoint_freq = max(1, num_epochs // 20)
    
    # New save and plot frequency logic:
    # Aim for at least 20 saves, but save at least every 100 epochs.
    base_freq = max(1, num_epochs // 20)
    save_and_plot_freq = min(base_freq, 100)
    print(f"Data and plots will be saved every {save_and_plot_freq} epochs.", flush=True)

    plot_loss_freq = save_and_plot_freq
    npz_save_freq = save_and_plot_freq
    
    vtk_out = VTKFile(str(output_dir / "ch_learn_adjoint.pvd"))
    use_wandb = not args.no_wandb
    
    # Training state
    preds_collection = []
    epochs_collection = []
    target_final_global = None
    all_epochs_comparison_data = []
    min_loss_weighted = float('inf')
    all_nn_outputs = []
    
    # Pre-allocate Firedrake functions to reuse across epochs (avoids memory leak)
    c_snapshots = [Function(V, name=f"c_snapshot_{i}") for i in range(num_timesteps)]
    eta_snapshots = [Function(V, name=f"eta_snapshot_{i}") for i in range(num_timesteps)]
    dfdc_outputs = [Function(V, name=f"dfdc_pred_{i}") for i in range(num_timesteps)]
    dfdeta_outputs = [Function(V, name=f"dfdeta_pred_{i}") for i in range(num_timesteps)]
    g_c_list = [Function(V) for _ in range(num_timesteps)]
    g_eta_list = [Function(V) for _ in range(num_timesteps)]
    
    # Resume NPZ data if needed
    npz_path = output_dir / "post_processing_data.npz"
    if start_epoch > 0 and npz_path.exists():
        print(f"Resuming from checkpoint, loading existing .npz data from {npz_path}", flush=True)
        with np.load(npz_path, allow_pickle=True) as data:
            preds_collection = list(data.get('preds_collection', []))
            epochs_collection = list(data.get('epochs_collection', []))
            target_final_global = data.get('target_final_global')
            all_epochs_comparison_data = list(data.get('all_epochs_comparison_data', []))
            all_nn_outputs = list(data.get('all_nn_outputs', []))
            
            epoch_losses_weighted = list(data.get('epoch_losses_weighted', []))
            epoch_losses_true = list(data.get('epoch_losses_true', []))
            epoch_losses_c_true = list(data.get('epoch_losses_c_true', []))
            epoch_losses_eta_weighted = list(data.get('epoch_losses_eta_weighted', []))
            epoch_losses_eta_true = list(data.get('epoch_losses_eta_true', []))
            epoch_losses_int_weighted = list(data.get('epoch_losses_int_weighted', []))
            epoch_losses_int_true = list(data.get('epoch_losses_int_true', []))
            epoch_numbers = list(data.get('epoch_numbers', []))

    if args.scheduler == 'plateau' and args.warmup_epochs > 0:
        warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda epoch: (epoch + 1) / args.warmup_epochs
        )
        if start_epoch > 0:
            for i in range(start_epoch):
                warmup_scheduler.step()
    else:
        warmup_scheduler = None

    print(f"Starting training for {num_epochs} epochs...", flush=True)
    
    for epoch in range(start_epoch, num_epochs):
        (loss_w, loss_t, loss_c_t, loss_eta_w, loss_eta_t, 
         loss_int_w, loss_int_t, elapsed_time, u_curr, processed_comparison_data) = train_epoch(
            epoch, num_epochs, model, optimizer, device, u_ic, u, c_target_list, eta_target_list,
            V, W, dt, M, lmbda, num_timesteps, vtk_out, ch_solver,
            c_snapshots, eta_snapshots, dfdc_outputs, dfdeta_outputs, g_c_list, g_eta_list,
            eta_loss_weight=args.eta_loss_weight, 
            integrability_weight=args.integrability_weight,
            integrability_start_epoch=args.integrability_start_epoch,
            use_wandb=use_wandb,
            should_collect_data=True
        )
        
        old_lr = optimizer.param_groups[0]['lr']
        if scheduler is not None:
            if args.scheduler == 'plateau':
                if epoch < args.warmup_epochs:
                    warmup_scheduler.step()
                else:
                    scheduler.step(loss_w)
            else:
                scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        lr_changed = scheduler is not None and current_lr != old_lr

        if lr_changed and args.scheduler != 'cosine':
            print(f"Learning rate updated to {current_lr:.6e}", flush=True)

        # Store losses and epoch numbers for plotting
        epoch_numbers.append(epoch + 1)
        epoch_losses_weighted.append(loss_w)
        epoch_losses_true.append(loss_t)
        epoch_losses_c_true.append(loss_c_t)
        epoch_losses_eta_weighted.append(loss_eta_w)
        epoch_losses_eta_true.append(loss_eta_t)
        epoch_losses_int_weighted.append(loss_int_w)
        epoch_losses_int_true.append(loss_int_t)

        # Logging
        if use_wandb:
            wandb.log({
                "loss_weighted": loss_w, 
                "loss_true": loss_t,
                "loss_c_true": loss_c_t, 
                "loss_eta_true": loss_eta_t, 
                "loss_int_true": loss_int_t, 
                "epoch": epoch
            })
            if scheduler is not None:
                 wandb.log({"learning_rate": current_lr, "epoch": epoch})

        if loss_w < min_loss_weighted:
            min_loss_weighted = loss_w
            print(f"Epoch {epoch+1}/{num_epochs} finished in {elapsed_time:.2f} s, J_w={loss_w:.6e} (c_t={loss_c_t:.6e}, η_w={loss_eta_w:.6e}, int_t={loss_int_t:.6e})", flush=True)
            print(f"New minimum weighted loss: {min_loss_weighted:.6e}", flush=True)
        
        # Checkpointing
        if (epoch + 1) % checkpoint_freq == 0 or epoch == num_epochs - 1:
            save_checkpoint(epoch, model, optimizer, scheduler, epoch_losses_weighted, epoch_numbers, output_dir, filename="ch_learn_dfdc.pth")
            
        # Plotting
        if (epoch + 1) % plot_loss_freq == 0 or epoch == num_epochs - 1:
            # Plot weighted losses (original)
            plot_loss_vs_epochs(epoch_numbers, epoch_losses_weighted, output_dir / "lve_dfdc.png", 
                                min_loss=min_loss_weighted, losses_c=epoch_losses_c_true, 
                                losses_eta=epoch_losses_eta_weighted,
                                losses_int=epoch_losses_int_weighted)
            
            # Plot true losses (new)
            plot_loss_vs_epochs(epoch_numbers, epoch_losses_true, output_dir / "lve_true.png", 
                                losses_c=epoch_losses_c_true, 
                                losses_eta=epoch_losses_eta_true,
                                losses_int=epoch_losses_int_true)

        # Store data as requested
        all_epochs_comparison_data.append({'epoch': epoch, 'data': processed_comparison_data})
        
        pred_c = u_curr.sub(0).dat.data_ro.copy().astype(np.float32)
        pred_eta = u_curr.sub(2).dat.data_ro.copy().astype(np.float32)
        preds_collection.append({'c': pred_c, 'eta': pred_eta})
        epochs_collection.append(epoch + 1)
        
        if target_final_global is None:
            target_c = c_target_list[-1].dat.data_ro.copy().astype(np.float32)
            target_eta = eta_target_list[-1].dat.data_ro.copy().astype(np.float32)
            target_final_global = {'c': target_c, 'eta': target_eta}
            
        # NPZ Save
        if (epoch + 1) % npz_save_freq == 0 or epoch == num_epochs - 1:
            save_npz_data(output_dir, epoch + 1, preds_collection, epochs_collection,
                          target_final_global, all_epochs_comparison_data,
                          epoch_losses_weighted, epoch_numbers, model, device, use_wandb, all_nn_outputs,
                          args.chi, args.chi_ac, args.N1, args.N2, args.Weta, args.z, args.z0,
                          args.T, args.dt, args.M, lmbda, args.lmbda_eta, args.L_kinetic,
                          mesh_coords=mesh_coords,
                          epoch_losses_weighted=epoch_losses_weighted,
                          epoch_losses_true=epoch_losses_true,
                          epoch_losses_c_true=epoch_losses_c_true,
                          epoch_losses_eta_weighted=epoch_losses_eta_weighted,
                          epoch_losses_eta_true=epoch_losses_eta_true,
                          epoch_losses_int_weighted=epoch_losses_int_weighted,
                          epoch_losses_int_true=epoch_losses_int_true)

    print("Training finished.", flush=True)
    if use_wandb:
        wandb.finish()

if __name__ == "__main__":
    main()
