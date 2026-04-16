import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from pathlib import Path
import argparse
import shutil
from scipy.integrate import cumulative_trapezoid

def create_4_plot_configs(base_name, label_suffix=""):
    return [
        {'label': f'Prediction {label_suffix}', 'visible_pattern': [True, False, False], 'filename': f'{base_name}_prediction_3d.html'},
        {'label': f'Target {label_suffix}', 'visible_pattern': [False, True, False], 'filename': f'{base_name}_target_3d.html'},
        {'label': f'Prediction and Target {label_suffix}', 'visible_pattern': [True, True, False], 'filename': f'{base_name}_prediction_and_target_3d.html'},
        {'label': f'Error {label_suffix}', 'visible_pattern': [False, False, True], 'filename': f'{base_name}_error_3d.html'}
    ]

def plot_phase_space_data_3d(c_lin, eta_lin, all_nn_outputs, target_values, ylabel, output_dir, base_name):
    """
    Generalized function to create the 4-plot set (Pred, Target, Combined, Error) 
    for data defined on the (c, eta) phase space.
    """
    C, ETA = np.meshgrid(c_lin, eta_lin)
    configs = create_4_plot_configs(base_name)
    
    # Pre-calculate error surfaces for each epoch
    all_errors = []
    for nn_output_data in all_nn_outputs:
        pred = nn_output_data['output_to_plot']
        all_errors.append(np.abs(pred - target_values))

    for config in configs:
        try:
            fig = go.Figure()
            for i, nn_output_data in enumerate(all_nn_outputs):
                epoch = nn_output_data['epoch']
                pred = nn_output_data['output_to_plot']
                err = all_errors[i]
                
                # Prediction
                fig.add_trace(go.Surface(x=C, y=ETA, z=pred, name=f'Pred (ep {epoch})', 
                                         colorscale='Viridis', showscale=False, visible=False))
                # Target
                fig.add_trace(go.Surface(x=C, y=ETA, z=target_values, name='Target', 
                                         colorscale=[[0, 'black'], [1, 'black']], showscale=False, opacity=0.4, visible=False))
                # Error
                fig.add_trace(go.Surface(x=C, y=ETA, z=err, name=f'Error (ep {epoch})', 
                                         colorscale='Reds', showscale=True, visible=False))

            steps = []
            for i, nn_output_data in enumerate(all_nn_outputs):
                visibility = [False] * len(fig.data)
                for trace_idx_in_epoch, is_visible in enumerate(config['visible_pattern']):
                    if is_visible:
                        visibility[i * 3 + trace_idx_in_epoch] = True
                
                steps.append(dict(method="update", label=str(nn_output_data['epoch']),
                                  args=[{"visible": visibility}, 
                                        {"title.text": f"{ylabel}: {config['label']} (Epoch {nn_output_data['epoch']})"}]))

            sliders = [dict(active=0, currentvalue={"prefix": "Epoch: "}, pad={"t": 50}, steps=steps)]
            
            initial_vis = [False] * len(fig.data)
            for trace_idx_in_epoch, is_visible in enumerate(config['visible_pattern']):
                if is_visible: initial_vis[trace_idx_in_epoch] = True
            for i in range(len(fig.data)): fig.data[i].visible = initial_vis[i]

            fig.update_layout(sliders=sliders, 
                              title_text=f"{ylabel}: {config['label']} (Epoch {all_nn_outputs[0]['epoch']})",
                              scene=dict(xaxis_title='c', yaxis_title='eta', zaxis_title=ylabel), 
                              template="plotly_white")
            
            pio.write_html(fig, output_dir / config['filename'])
            print(f"Saved {ylabel} phase plot to {config['filename']}")
        except Exception as e:
            print(f"Could not create {ylabel} phase plot for {config['label']}: {e}")

def reproduce_plots(npz_path):
    npz_path = Path(npz_path)
    if not npz_path.is_file(): return
    plot_output_dir = npz_path.parent / "reproduced_plots"
    if plot_output_dir.exists(): shutil.rmtree(plot_output_dir)
    plot_output_dir.mkdir()
    data = np.load(npz_path, allow_pickle=True)

    if 'c_values_nn' in data and 'all_nn_outputs' in data:
        c_lin = data['c_values_nn']
        eta_lin = data.get('eta_values_nn', np.array([0.0])) 
        all_outputs_raw = data['all_nn_outputs']
        
        # Physical constants
        chi_aa = float(data.get('chi', 1.0))
        chi_ac = float(data.get('chi_ac', 1.0))
        N1 = float(data.get('N1', 4.0))
        N2 = float(data.get('N2', 4.0))
        Weta = float(data.get('Weta', 1.0))
        z = float(data.get('z', 0.5))
        z0 = float(data.get('z0', 1.0))

        C, ETA = np.meshgrid(c_lin, eta_lin)
        epsilon = 1e-10
        c_safe = np.clip(C, epsilon, 1 - epsilon)
        
        # --- df/dc ---
        f_cr_val = Weta*(ETA**4/4 - (z+z0)*ETA**3/3 + z*z0*ETA**2/2)
        true_dfdc = (np.log(c_safe)/N1 + 1/N1) - (np.log(1-c_safe)/N2 + 1/N2) + \
                    chi_aa*(1 - 2*c_safe) + f_cr_val + chi_ac*(1 - 2*c_safe)
        
        outputs_dfdc = []
        for d in all_outputs_raw:
            new_d = d.copy()
            new_d['output_to_plot'] = d['output'][:, :, 0]
            outputs_dfdc.append(new_d)
        plot_phase_space_data_3d(c_lin, eta_lin, outputs_dfdc, true_dfdc, "df_dc", plot_output_dir, "dfdc")

        # --- df/deta ---
        df_cr_deta = Weta*(ETA**3 - (z+z0)*ETA**2 + z*z0*ETA)
        true_dfdeta = c_safe * df_cr_deta
        
        outputs_dfdeta = []
        for d in all_outputs_raw:
            new_d = d.copy()
            new_d['output_to_plot'] = d['output'][:, :, 1]
            outputs_dfdeta.append(new_d)
        plot_phase_space_data_3d(c_lin, eta_lin, outputs_dfdeta, true_dfdeta, "df_deta", plot_output_dir, "dfdeta")

        # --- Free Energy (f) ---
        f_mix = (c_safe * np.log(c_safe) / N1) + ((1 - c_safe) * np.log(1 - c_safe) / N2) + chi_aa * c_safe * (1 - c_safe)
        f_cp = chi_ac * c_safe * (1 - c_safe)
        true_f = f_mix + c_safe * f_cr_val + f_cp
        true_f = true_f - true_f[0,0]

        outputs_f = []
        for d in all_outputs_raw:
            dfdc_grid = d['output'][:, :, 0]
            dfdeta_grid = d['output'][:, :, 1]
            f_c0 = cumulative_trapezoid(dfdc_grid[0, :], c_lin, initial=0)
            f_grid = np.zeros_like(dfdc_grid)
            for j in range(len(c_lin)):
                f_column = cumulative_trapezoid(dfdeta_grid[:, j], eta_lin, initial=0)
                f_grid[:, j] = f_column + f_c0[j]
            new_d = d.copy()
            new_d['output_to_plot'] = f_grid
            outputs_f.append(new_d)
        plot_phase_space_data_3d(c_lin, eta_lin, outputs_f, true_f, "Free Energy", plot_output_dir, "free_energy")

    # --- Simulation Plots (Spatial) ---
    if 'all_epochs_comparison_data' in data:
        all_data = data['all_epochs_comparison_data']
        mesh_coords = data.get('mesh_coords')
        if len(all_data) > 200:
            all_data = [all_data[i] for i in np.linspace(0, len(all_data)-1, 200, dtype=int)]
        plot_simulation_data_3d(all_data, plot_output_dir / "sim_c", 'c', mesh_coords=mesh_coords)
        plot_simulation_data_3d(all_data, plot_output_dir / "sim_eta", 'eta', mesh_coords=mesh_coords)

def plot_simulation_data_3d(all_epochs_data, output_path, field='c', mesh_coords=None):
    """
    Creates interactive 3D Plotly plots of the simulation data.
    """
    if len(all_epochs_data) == 0: return
    output_dir = output_path.parent
    base_name = f"simulation_{field}"
    configs = create_4_plot_configs(base_name, label_suffix=f"({field})")

    for config in configs:
        try:
            fig = go.Figure()
            first_epoch_data = all_epochs_data[0]['data']
            pred_idx, targ_idx = (1, 2) if field == 'c' else (3, 4)
            num_dofs = first_epoch_data[0][pred_idx].size
            
            if mesh_coords is not None:
                x_coords = mesh_coords[:, 0]
                xaxis_title = 'X Coordinate'
            else:
                x_coords = np.arange(num_dofs)
                xaxis_title = 'DOF index'

            for epoch_data in all_epochs_data:
                t_coords = np.array([d[0] for d in epoch_data['data']])
                C_pred = np.array([d[pred_idx] for d in epoch_data['data']])
                C_targ = np.array([d[targ_idx] for d in epoch_data['data']])
                C_err = np.abs(C_pred - C_targ)
                X, T = np.meshgrid(x_coords, t_coords)
                fig.add_trace(go.Surface(z=C_pred, x=X, y=T, name='Prediction', colorscale='Viridis', showscale=False, visible=False))
                fig.add_trace(go.Surface(z=C_targ, x=X, y=T, name='Target', colorscale=[[0, "blue"], [1, "blue"]], opacity=0.4, showscale=False, visible=False))
                fig.add_trace(go.Surface(z=C_err, x=X, y=T, name='Error', colorscale='Reds', showscale=True, visible=False))

            steps = []
            for i, epoch_data in enumerate(all_epochs_data):
                visibility = [False] * len(fig.data)
                for t_idx, is_vis in enumerate(config['visible_pattern']):
                    if is_vis: visibility[i*3 + t_idx] = True
                steps.append(dict(method="update", label=str(epoch_data['epoch']),
                                  args=[{"visible": visibility}, {"title": f"{field.upper()}: {config['label']} (Epoch {epoch_data['epoch']})"}]))

            sliders = [dict(active=0, currentvalue={"prefix": "Epoch: "}, pad={"t": 50}, steps=steps)]
            initial_vis = [False] * len(fig.data)
            for t_idx, is_vis in enumerate(config['visible_pattern']):
                if is_vis: initial_vis[t_idx] = True
            for k in range(len(fig.data)): fig.data[k].visible = initial_vis[k]

            fig.update_layout(sliders=sliders, title_text=f"{field.upper()} Simulation: {config['label']} (Epoch {all_epochs_data[0]['epoch']})",
                              scene=dict(xaxis_title=xaxis_title, yaxis_title='Timestep', zaxis_title=field), template="plotly_white")
            pio.write_html(fig, output_dir / config['filename'])
            print(f"Saved {field} 3D plot to {config['filename']}")
        except Exception as e:
            print(f"Could not create {field} 3D plot: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reproduce plots from CH-AC .npz output.")
    parser.add_argument("npz_file", type=str, help="Path to the post_processing_data.npz file.")
    args = parser.parse_args()
    reproduce_plots(args.npz_file)
