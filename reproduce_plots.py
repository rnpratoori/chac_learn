import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from pathlib import Path
import argparse
import shutil

def plot_nn_output_animation_3d(c_lin, eta_lin, all_nn_outputs, ylabel, output_path, output_idx=0, 
                                chi_aa=1.0, chi_ac=1.0, N1=4.0, N2=4.0, Weta=1.0, z=0.5, z0=1.0):
    """
    Creates an animated Plotly 3D surface plot of the neural network output vs. c and eta.
    """
    try:
        fig = go.Figure()
        
        # Calculate true values on the 2D grid
        C, ETA = np.meshgrid(c_lin, eta_lin)
        epsilon = 1e-10
        c_safe = np.clip(C, epsilon, 1 - epsilon)
        
        f_cr_val = Weta*(ETA**4/4 - (z+z0)*ETA**3/3 + z*z0*ETA**2/2)
        df_cr_deta = Weta*(ETA**3 - (z+z0)*ETA**2 + z*z0*ETA)
        
        if output_idx == 0: # df/dc
            true_values = (np.log(c_safe)/N1 + 1/N1) - (np.log(1-c_safe)/N2 + 1/N2) + \
                          chi_aa*(1 - 2*c_safe) + f_cr_val + chi_ac*(1 - 2*c_safe)
            label_name = "True df/dc"
        else: # df/deta
            true_values = c_safe * df_cr_deta
            label_name = "True df/deta"

        # Reference surface (dashed style not supported for surface, so we'll use a specific color)
        fig.add_trace(go.Surface(x=C, y=ETA, z=true_values, name=label_name, 
                                 colorscale=[[0, 'black'], [1, 'black']], showscale=False, opacity=0.3))

        for nn_output_data in all_nn_outputs:
            epoch = nn_output_data['epoch']
            nn_output_values = nn_output_data['output'][:, :, output_idx]
            fig.add_trace(go.Surface(x=C, y=ETA, z=nn_output_values, name=f"Epoch {epoch}", visible=False))

        if len(fig.data) > 1:
            fig.data[1].visible = True

        steps = []
        for i, nn_output_data in enumerate(all_nn_outputs):
            epoch = nn_output_data['epoch']
            visibility = [True] + [False] * len(all_nn_outputs)
            visibility[i+1] = True
            steps.append(dict(method="update", label=str(epoch),
                              args=[{"visible": visibility}, {"title": f"Learned {ylabel} vs. (c, eta) (Epoch {epoch})"}]))

        sliders = [dict(active=0, currentvalue={"prefix": "Epoch: "}, pad={"t": 50}, steps=steps)]
        fig.update_layout(sliders=sliders, title=f"Learned {ylabel} vs. (c, eta) (Epoch {all_nn_outputs[0]['epoch']})",
                          scene=dict(xaxis_title="Concentration (c)", yaxis_title="Crystallinity (eta)", zaxis_title=ylabel), 
                          template="plotly_white")

        pio.write_html(fig, output_path)
        print(f"Saved {ylabel} 3D animation to {output_path}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Could not create {ylabel} 3D animation: {e}")

def plot_simulation_data_3d(all_epochs_data, output_path, field='c', mesh_coords=None):
    """
    Creates interactive 3D Plotly plots of the simulation data.
    """
    if len(all_epochs_data) == 0: return

    output_dir = output_path.parent
    base_name = f"simulation_{field}"

    plot_configs = [
        {'label': 'Prediction', 'visible_pattern': [True, False, False], 'filename': f'{base_name}_prediction_3d.html'},
        {'label': 'Target', 'visible_pattern': [False, True, False], 'filename': f'{base_name}_target_3d.html'},
        {'label': 'Prediction and Target', 'visible_pattern': [True, True, False], 'filename': f'{base_name}_prediction_and_target_3d.html'},
        {'label': 'Error', 'visible_pattern': [False, False, True], 'filename': f'{base_name}_error_3d.html'}
    ]

    for config in plot_configs:
        try:
            fig = go.Figure()
            first_epoch_data = all_epochs_data[0]['data']
            pred_idx, targ_idx = (1, 2) if field == 'c' else (3, 4)
            num_sim_timesteps = len(first_epoch_data)
            num_dofs = first_epoch_data[0][pred_idx].size
            
            if mesh_coords is not None:
                # Use the first coordinate (x) for the plot axis
                # In 3D, this is a projection/slice representation
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
                fig.add_trace(go.Surface(z=C_pred, x=X, y=T, name='Prediction', colorscale=[[0, "red"], [1, "red"]], showscale=False, visible=False))
                fig.add_trace(go.Surface(z=C_targ, x=X, y=T, name='Target', colorscale=[[0, "blue"], [1, "blue"]], showscale=False, visible=False))
                fig.add_trace(go.Surface(z=C_err, x=X, y=T, name='Error', colorscale='Bluered', cmin=0, cmax=1, showscale=True, visible=False))

            steps = []
            for i, epoch_data in enumerate(all_epochs_data):
                visibility = [False] * len(fig.data)
                for trace_idx, is_visible in enumerate(config['visible_pattern']):
                    if is_visible: visibility[i * 3 + trace_idx] = True
                steps.append(dict(method="update", label=str(epoch_data['epoch']),
                                  args=[{"visible": visibility}, {"title.text": f"{field.upper()} Simulation: {config['label']} (Epoch {epoch_data['epoch']})"}]))

            sliders = [dict(active=0, currentvalue={"prefix": "Epoch: "}, pad={"t": 50}, steps=steps)]
            initial_visibility = [False] * len(fig.data)
            for trace_idx in [idx for idx, is_vis in enumerate(config['visible_pattern']) if is_vis]: initial_visibility[trace_idx] = True
            for i in range(len(fig.data)): fig.data[i].visible = initial_visibility[i]

            fig.update_layout(sliders=sliders, title_text=f"{field.upper()} Simulation: {config['label']} (Epoch {all_epochs_data[0]['epoch']})",
                              scene=dict(xaxis_title=xaxis_title, yaxis_title='Timestep', zaxis_title=field), template="plotly_white")
            pio.write_html(fig, output_dir / config['filename'])
            print(f"Saved {field} 3D plot to {config['filename']}")
        except Exception as e:
            print(f"Could not create {field} 3D plot for {config['label']}: {e}")

def reproduce_plots(npz_path):
    npz_path = Path(npz_path)
    if not npz_path.is_file(): return
    plot_output_dir = npz_path.parent / "reproduced_plots"
    if plot_output_dir.exists(): shutil.rmtree(plot_output_dir)
    plot_output_dir.mkdir()
    data = np.load(npz_path, allow_pickle=True)

    if 'c_values_nn' in data and 'all_nn_outputs' in data:
        c_lin = data['c_values_nn']
        # Handle cases where eta_values_nn might be missing in older NPZ files
        eta_lin = data.get('eta_values_nn', np.array([0.0])) 
        all_outputs = data['all_nn_outputs']
        
        # Extract physical constants for reference solution
        params = {
            'chi_aa': float(data.get('chi', 1.0)),
            'chi_ac': float(data.get('chi_ac', 1.0)),
            'N1': float(data.get('N1', 4.0)),
            'N2': float(data.get('N2', 4.0)),
            'Weta': float(data.get('Weta', 1.0)),
            'z': float(data.get('z', 0.5)),
            'z0': float(data.get('z0', 1.0))
        }

        plot_nn_output_animation_3d(c_lin, eta_lin, all_outputs, "df_dc", plot_output_dir / "dfdc_animation_3d.html", 0, **params)
        plot_nn_output_animation_3d(c_lin, eta_lin, all_outputs, "df_deta", plot_output_dir / "dfdeta_animation_3d.html", 1, **params)

    if 'all_epochs_comparison_data' in data:
        all_data = data['all_epochs_comparison_data']
        mesh_coords = data.get('mesh_coords')
        
        if len(all_data) > 500:
            all_data = [all_data[i] for i in np.linspace(0, len(all_data)-1, 500, dtype=int)]
        plot_simulation_data_3d(all_data, plot_output_dir / "sim_c", 'c', mesh_coords=mesh_coords)
        plot_simulation_data_3d(all_data, plot_output_dir / "sim_eta", 'eta', mesh_coords=mesh_coords)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reproduce plots from CH-AC .npz output.")
    parser.add_argument("npz_file", type=str, help="Path to the post_processing_data.npz file.")
    args = parser.parse_args()
    reproduce_plots(args.npz_file)
