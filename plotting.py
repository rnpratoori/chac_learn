import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import torch

def plot_nn_output_vs_c(net, device, ylabel, title, eta_val=0.0, output_idx=0):
    """
    Plots the output of a given neural network against the concentration (c) for a fixed eta.
    Returns the Plotly figure.
    """
    try:
        c_values = np.linspace(0, 1, 200).reshape(-1, 1)
        eta_values = np.full_like(c_values, eta_val)
        input_tensor = torch.from_numpy(np.hstack([c_values, eta_values])).to(device)
        
        with torch.no_grad():
            output_values = net(input_tensor).cpu().numpy()[:, output_idx]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=c_values.flatten(), y=output_values.flatten(), mode='lines'))
        fig.update_layout(
            title=f"{title} (at eta={eta_val})",
            xaxis_title="Concentration (c)",
            yaxis_title=ylabel,
            template="plotly_white"
        )
        return fig
    except Exception as e:
        print(f"Could not create nn output vs c plot: {e}")
        return None


def plot_combined_final_timestep(preds_collection, epochs_collection, target_final_global, field='c'):
    """
    Creates a combined plot showing the final-timestep predictions from several epochs against the ground truth.
    field: 'c' or 'eta'
    """
    if len(preds_collection) > 0:
        try:
            fig = go.Figure()
            # Extract data based on field
            if isinstance(preds_collection[0], dict):
                first_pred = preds_collection[0][field]
            else:
                first_pred = preds_collection[0]
                
            x = np.arange(first_pred.size)
            
            for item, ep in zip(preds_collection, epochs_collection):
                arr = item[field] if isinstance(item, dict) else item
                fig.add_trace(go.Scatter(x=x, y=arr, mode='lines', name=f'Pred (ep {ep})', line=dict(width=1), opacity=0.9))

            if target_final_global is not None:
                targ_arr = target_final_global[field] if isinstance(target_final_global, dict) else target_final_global
                fig.add_trace(go.Scatter(x=x, y=targ_arr, mode='lines', name='Ground truth (final time)', line=dict(color='black', width=2)))

            fig.update_layout(
                title=f"Final timestep ({field}): predictions (multiple epochs) vs ground truth",
                xaxis_title="DOF index",
                yaxis_title=field,
                template="plotly_white",
                legend=dict(font=dict(size=10))
            )
            return fig
        except Exception as e:
            print(f"Could not create combined final-timestep plot for {field}: {e}")
            return None
    return None


def plot_loss_vs_epochs(epochs, losses, output_path, min_loss=None, losses_c=None, losses_eta=None, losses_int=None):
    """
    Plots the training loss against epochs using Matplotlib and saves as PNG.
    """
    try:
        fig, ax = plt.subplots()
        ax.plot(epochs, losses, '-', label='Total Loss', linewidth=2)
        
        if losses_c is not None and len(losses_c) == len(epochs):
            ax.plot(epochs, losses_c, '--', label='Loss C', alpha=0.7)
            
        if losses_eta is not None and len(losses_eta) == len(epochs):
            ax.plot(epochs, losses_eta, ':', label='Loss Eta', alpha=0.7)

        if losses_int is not None and len(losses_int) == len(epochs):
            ax.plot(epochs, losses_int, '-.', label='Loss Integrability', alpha=0.7)
            
        if min_loss is not None:
            ax.axhline(y=min_loss, color='r', linestyle='--', label=f"Min Total Loss: {min_loss:.6e}", linewidth=1)
            
        ax.set(xlabel="Epoch", ylabel="Loss (log scale)", title="Loss vs. Epochs")
        ax.set_yscale('log')
        ax.grid(True, which="both", ls="-", alpha=0.2)
        ax.legend()
        
        # Ensure extension is .png
        path = str(output_path)
        if not path.endswith('.png'):
            path = path.rsplit('.', 1)[0] + '.png'
        
        fig.savefig(path)
        plt.close(fig)
        return fig
    except Exception as e:
        print(f"Could not create loss vs epochs plot: {e}")
        return None


def plot_multi_timestep_comparison_2d(epoch, comparison_data, field='c', title=None):
    """
    Creates a 2D Plotly figure with predictions and targets at multiple timesteps.
    field: 'c' or 'eta'
    """
    if not comparison_data:
        return None
    
    fig = go.Figure()
    colors = px.colors.qualitative.Plotly
    
    # Check if comparison_data has eta (5-tuple instead of 3-tuple)
    has_eta = len(comparison_data[0]) == 5
    
    for i, data in enumerate(comparison_data):
        if has_eta:
            timestep, c_pred, c_targ, eta_pred, eta_targ = data
            pred_np = c_pred if field == 'c' else eta_pred
            target_np = c_targ if field == 'c' else eta_targ
        else:
            timestep, pred_np, target_np = data
            
        x = np.arange(pred_np.size)
        color = colors[i % len(colors)]
        
        fig.add_trace(go.Scatter(x=x, y=pred_np, mode='lines', 
                                 name=f'Pred (t={timestep + 1})', 
                                 line=dict(color=color)))
        fig.add_trace(go.Scatter(x=x, y=target_np, mode='lines', 
                                 name=f'Targ (t={timestep + 1})', 
                                 line=dict(color=color, dash='dash')))

    fig.update_layout(
        title=title or f"Epoch {epoch} - 2D Multi-timestep Comparison ({field})",
        xaxis_title="DOF index",
        yaxis_title=field,
        template="plotly_white",
        legend=dict(font=dict(size=8), orientation="h")
    )
    return fig


def plot_multi_timestep_comparison_3d(epoch, comparison_data, field='c', title=None):
    """
    Creates a 3D Plotly figure with predictions and targets as surfaces.
    field: 'c' or 'eta'
    """
    if not comparison_data:
        return None

    # Check if comparison_data has eta
    has_eta = len(comparison_data[0]) == 5
    
    if has_eta:
        x_coords = np.arange(comparison_data[0][1].size if field == 'c' else comparison_data[0][3].size)
    else:
        x_coords = np.arange(comparison_data[0][1].size)
        
    t_coords = np.array([d[0] for d in comparison_data])
    
    if has_eta:
        C_pred = np.array([d[1] if field == 'c' else d[3] for d in comparison_data])
        C_targ = np.array([d[2] if field == 'c' else d[4] for d in comparison_data])
    else:
        C_pred = np.array([d[1] for d in comparison_data])
        C_targ = np.array([d[2] for d in comparison_data])

    fig = go.Figure()

    # Prediction surface
    fig.add_trace(go.Surface(x=x_coords, y=t_coords, z=C_pred, 
                             name='Prediction', colorscale='Viridis', showscale=False, opacity=0.8))
    
    # Target surface
    fig.add_trace(go.Surface(x=x_coords, y=t_coords, z=C_targ, 
                             name='Target', colorscale='Hot', showscale=False, opacity=0.6))

    fig.update_layout(
        title=title or f"Epoch {epoch} - 3D Space-Time Comparison ({field})",
        scene=dict(
            xaxis_title='DOF Index',
            yaxis_title='Timestep',
            zaxis_title=field
        ),
        template="plotly_white",
        margin=dict(l=0, r=0, b=0, t=40)
    )
    return fig