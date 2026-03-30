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


# plot_combined_final_timestep removed — uses DOF index as x-axis, meaningless in 3D.


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


# plot_multi_timestep_comparison_2d commented out — DOF-index x-axis is 1D-specific;
# no direct 3D replacement for now.
#
# def plot_multi_timestep_comparison_2d(epoch, comparison_data, field='c', title=None):
#     ...


# plot_multi_timestep_comparison_3d commented out — (DOF index, timestep, value) surface
# is only meaningful in 1D; no direct 3D replacement for now.
#
# def plot_multi_timestep_comparison_3d(epoch, comparison_data, field='c', title=None):
#     ...