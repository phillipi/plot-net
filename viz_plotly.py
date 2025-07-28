import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import viz_matplotlib as viz

def viz_static(net, embeddings, embeddings_nonsequential, Y, grid, args, out_path="viz_mapping.html"):
    n_layers = len(embeddings) - 1
    n_points = embeddings[0].shape[0]

    if Y is None:
        Y = np.zeros(n_points, dtype=int)
    
    if net.d == 2:
        class_colors = viz.my_colors_2
    else:
        class_colors = viz.my_colors_3

    # Create the 3D figure
    fig = go.Figure()

    z_step = 3.2

    # Add axes box for each layer
    for l in range(n_layers + 1):
        z = l * z_step
        
        # Add grid lines
        for x in [-1, 0, 1]:
            fig.add_trace(go.Scatter3d(
                x=[x, x], y=[-1, 1], z=[z, z],
                mode='lines',
                line=dict(color='gray', width=3),
                showlegend=False,
                hoverinfo='skip'
            ))
        
        for y in [-1, 0, 1]:
            fig.add_trace(go.Scatter3d(
                x=[-1, 1], y=[y, y], z=[z, z],
                mode='lines',
                line=dict(color='gray', width=3),
                showlegend=False,
                hoverinfo='skip'
            ))
        
        # add grid surface
        fig.add_trace(go.Mesh3d(
            x=[-1, 1, 1, -1],
            y=[-1, -1, 1, 1],
            z=[z, z, z, z],
            i=[0, 0], j=[1, 2], k=[2, 3],
            color='white',
            opacity=0.3,
            flatshading=True,          # Ensures it's drawn as a flat plane
            lighting=dict(ambient=1),  # Prevent shadowing effects
            hoverinfo='skip',
            showscale=False
        ))
        
        # Add wireframe for transformed grid
        if l < len(embeddings_nonsequential):
            grid_data = embeddings_nonsequential[l]
            if len(grid_data) >= 4:  # Need at least 4 points for a 2x2 grid
                # Reshape to 2x2 grid
                grid_reshaped = grid_data[:4].reshape(2, 2, 2)
                
                # Add wireframe lines
                for i in range(2):
                    for j in range(2):
                        # Horizontal lines
                        if i < 1:
                            fig.add_trace(go.Scatter3d(
                                x=[grid_reshaped[i,j,0], grid_reshaped[i+1,j,0]],
                                y=[-grid_reshaped[i,j,1], -grid_reshaped[i+1,j,1]],
                                z=[z, z],
                                mode='lines',
                                line=dict(color='rgb(0.5,0.5,0.5)', width=5, dash='dash'),
                                showlegend=False,
                                hoverinfo='skip'
                            ))
                        # Vertical lines
                        if j < 1:
                            fig.add_trace(go.Scatter3d(
                                x=[grid_reshaped[i,j,0], grid_reshaped[i,j+1,0]],
                                y=[-grid_reshaped[i,j,1], -grid_reshaped[i,j+1,1]],
                                z=[z, z],
                                mode='lines',
                                line=dict(color='rgb(0.5,0.5,0.5)', width=5, dash='dash'),
                                showlegend=False,
                                hoverinfo='skip'
                            ))

    # Add mapping lines for grid
    for l in range(n_layers):
        z0 = l * z_step
        z1 = (l + 1) * z_step
        
        for i in range(grid.shape[0]):
            x0, y0 = grid[i]
            x1, y1 = embeddings_nonsequential[l+1][i]
            
            fig.add_trace(go.Scatter3d(
                x=[x0, x1], y=[-y0, -y1], z=[z0, z1],
                mode='lines+markers',
                line=dict(color='rgb(0.2,0.2,0.2)', width=5, dash='dash'),
                marker=dict(size=3, color='rgb(0.2,0.2,0.2)'),
                showlegend=False,
                hoverinfo='skip'
            ))

    # Add data points and mappings between them
    for l in range(n_layers + 1):
        z = l * z_step
        
        # Add data points
        x_coords = embeddings[l][:, 0]
        y_coords = -embeddings[l][:, 1]
        z_coords = [z] * n_points
        
        # Color points by class
        colors = [class_colors[Y[i] % len(class_colors)] for i in range(n_points)]
        
        fig.add_trace(go.Scatter3d(
            x=x_coords, y=y_coords, z=z_coords,
            mode='markers',
            marker=dict(
                size=3,
                color=colors,
                opacity=0.5,
                line=dict(color='black', width=0)
            ),
            name=f'Layer {l}',
            hoverinfo='skip'
        ))

        # Add mappings between layers
        if l < n_layers:
            for i in range(n_points):
                x1, y1 = embeddings[l+1][i]
                rgb_color = class_colors[Y[i]]
                color = f'rgb({int(rgb_color[0]*255)}, {int(rgb_color[1]*255)}, {int(rgb_color[2]*255)})'
                fig.add_trace(go.Scatter3d(
                    x=[embeddings[l][i, 0], x1],
                    y=[-embeddings[l][i, 1], -y1],
                    z=[z, z + z_step],
                    mode='lines',
                    line=dict(
                        color=color,
                        width=2,
                        dash='dot'
                    ),
                    showlegend=False,
                    hoverinfo='skip'
                ))

    # Update layout
    zoom_out = 1+np.maximum(n_layers, 1)  # Increase this to zoom out more
    fig.update_layout(
        title='3D Neural Network Visualization',
        hovermode=False,
        scene=dict(
            camera=dict(
                eye=dict(x=1, y=1, z=1),  # eye doesn't affect orthographic much
                center=dict(x=0, y=0, z=0),
                projection=dict(type='orthographic')
            ),
            xaxis=dict(
                backgroundcolor='white',
                showbackground=True,
                range=[-zoom_out, zoom_out],
                autorange=False,
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                showspikes=False,
                title='',
            ),
            yaxis=dict(
                backgroundcolor='white',
                showbackground=True,
                range=[-zoom_out, zoom_out],
                autorange=False,
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                showspikes=False,
                title='',
            ),
            zaxis=dict(
                backgroundcolor='white',
                showbackground=True,
                range=[0, n_layers * z_step],
                autorange=False,
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                showspikes=False,
                title='',
            ),
            aspectmode='manual',
            aspectratio=dict(
                x=1, y=1, z=(n_layers * z_step) / (2 * zoom_out)
            ),
            bgcolor='white'
        ),
        margin=dict(l=0, r=0, t=50, b=0),
        showlegend=True
    )

    # Save as interactive HTML
    fig.write_html(out_path)
    print(f"Saved interactive 3D visualization to: {out_path}")

