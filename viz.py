import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.font_manager as font_manager
from matplotlib.animation import FuncAnimation
import subprocess

matplotlib.use('Agg')

def vizMapping(net, embeddings, embeddings_nonsequential, Y, grid_us, grid_vs, grid_flat, loss, z_step, ax):
    """
    Visualize how the network transforms the input data distribution layer by layer.

    Args:
        embeddings:[x, f1(x), f2(f1(x)), ...], where fi is layer i, (Nlayers x Nsamples x Nfeats)
        embeddings_nonsequential: [x, f1(x), f2(x), ...], where fi is layer i, (Nlayers x Nsamples x Nfeats)
        grid_us: Grid of u values
        grid_vs: Grid of v values
        grid_flat: Flattened grid of inputs
        loss: Loss value (optional)
        z_offset: Offset for the z-axis
        ax: Axis to plot on
    """

    min_u, max_u, min_v, max_v = -1, 1, -1, 1 

    n_layers = net.get_n_layers
    n_classes = len(np.unique(Y))

    for l in range(n_layers+1):
        # plot grid
        ax.plot_wireframe(np.reshape(embeddings_nonsequential[l,:,0],[2,2]), np.reshape(-embeddings_nonsequential[l,:,1],[2,2]), (l*z_step)*np.ones_like(np.reshape(embeddings_nonsequential[l,:,0],[2,2])), rcount=1, ccount=1, color = [0.7,0.7,0.7], linewidth=3.0, linestyle='--', zorder=2*l+0.0001)
        ax.plot_surface(grid_us, grid_vs, (l*z_step)*np.ones_like(grid_us), rcount=1, ccount=1, color = [1.0,1.0,1.0], shade=False, linewidth=0, zorder=2*l+0.001,alpha=0.75)
        ax.plot_wireframe(grid_us, grid_vs, (l*z_step)*np.ones_like(grid_us), rcount=1, ccount=1, color = [0.3,0.3,0.3], linewidth=6.0, zorder=2*l+0.01)
        ax.plot([min_u,max_u],[0,0],[l*z_step,l*z_step], color=[0.5,0.5,0.5], linewidth=4.0, zorder=2*l)
        ax.plot([0,0],[min_v,max_v],[l*z_step,l*z_step], color=[0.5,0.5,0.5], linewidth=4.0, zorder=2*l)

        # plot grid mapping
        if l<(n_layers):
            for i in range(0,embeddings_nonsequential.shape[1],1):
                ax.plot([grid_flat[i,0], embeddings_nonsequential[l+1,i,0]], [-grid_flat[i,1], -embeddings_nonsequential[l+1,i,1]], [l*z_step, (l+1)*z_step], '--', marker = 'o', color = [0.7, 0.7, 0.7], linewidth=3, markersize=10, zorder=2*l+1)#, alpha=.1)
                ax.scatter(embeddings_nonsequential[l+1,i,0], -embeddings_nonsequential[l+1,i,1], (l+1)*z_step, marker = '>', color = [0.1, 0.1, 0.1], s=80, zorder=2*l+1.01)

            # display layer name
            font_prop = font_manager.FontProperties(size=42)
            ax.text(-z_step, 0, (l+1)*z_step, net.get_layer_name(l), fontproperties=font_prop, bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.3'), zorder=1000)

        # plot data
        for i in range(embeddings.shape[1]):
            if Y is None:
                color = np.array([255, 0, 0, 255])/255
            else:
              if Y[i]==0:
                  color = np.array([255, 0, 0, 255])/255
              else:
                  color = np.array([0, 0, 255, 255])/255

            if l<n_layers:
                ax.plot([embeddings[l,i,0], embeddings[l+1,i,0]], [-embeddings[l,i,1], -embeddings[l+1,i,1]], [l*z_step, (l+1)*z_step], '--', marker = '.', color=color, linewidth=3, markersize=0, zorder=2*l+1, alpha=.5)
            ax.scatter(embeddings[l,i,0].ravel(), -embeddings[l,i,1].ravel(), (l)*z_step, color=color, s=160, alpha=0.5, linewidth=0, zorder=2*l+1.01)

    # display loss
    if loss is not None:
        font_prop = font_manager.FontProperties(size=42)
        ax.text(0, 0, (n_layers+0.75)*z_step, 'Loss: {:.2f}'.format(loss), fontproperties=font_prop, bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.3'), ha='center', zorder=1000)

    # Set axis limits tightly around the net and keep axes equal
    z_min, z_max = 0, n_layers*z_step
    mid_z = (z_max + z_min) / 2.0
    half_range = (z_max - z_min) / 2.0
    half_range = np.maximum(half_range, z_step*3)
    ax.set_xlim3d(0 - half_range/2, 0 + half_range/2)
    ax.set_ylim3d(0 - half_range/2, 0 + half_range/2)
    ax.set_zlim3d(mid_z - half_range/2, mid_z + half_range/2)
    ax.set_axis_off()


def viz_movie_init():
    return []


def viz_movie_update(frame, net, embeddings, embeddings_nonsequential, Y, grid_us, grid_vs, grid_flat, losses, z_step, ax):
    
    print('rendering frame {} of {}'.format(frame, embeddings.shape[0]))

    ax.cla()  # clear current frame
    ax.view_init(elev=35, azim=135)
    ax.dist = 8

    vizMapping(net, embeddings[frame], embeddings_nonsequential[frame], Y, grid_us, grid_vs, grid_flat, losses[frame], z_step, ax)

    return []


def viz_movie(net, embeddings, embeddings_nonsequential, Y, grid_us, grid_vs, grid_flat, losses):
    """
    Visualize the embeddings of a neural net over the course of training, at multiple checkpoints.
    Args:
        embeddings: (Niters x Nlayers x Nsamples x Nfeats)

    Outputs a movie of the embeddings.
    """

    # setup viz
    n_layers = net.get_n_layers
    z_step = 3.2
    fig = plt.figure()
    fig.set_size_inches(48, 48, forward=True)
    ax = fig.add_subplot(111, projection='3d', proj_type='ortho', computed_zorder=False)
    ax.view_init(elev=35,azim=135)
    ax.dist=8

    # visualize all the embeddings at each step of optimization
    ani = FuncAnimation(
        fig,
        viz_movie_update,
        frames=embeddings.shape[0],
        fargs=(net, embeddings, embeddings_nonsequential, Y, grid_us, grid_vs, grid_flat, losses, z_step, ax),
        init_func=viz_movie_init,
        blit=False
    )
    ani.save('./animation.gif', fps=60)

    # Trim whitespace and add a white border to the animation using ImageMagick
    import subprocess
    try:
        subprocess.run([
            'magick',
            'animation.gif',
            '-coalesce',
            '-trim',
            '-bordercolor', 'white',
            '-border', '100x100',
            '-repage', '0x0',
            '-layers', 'Optimize',
            'animation_cropped.gif'
        ], check=True)
    except Exception as e:
        print(f'Warning: Could not run magick to trim animation: {e}')


def viz_static(net, embeddings, embeddings_nonsequential, Y, grid_us, grid_vs, grid_flat):
    """
    Visualize the embeddings of a neural net at one checkpoint.

    Args:
        embeddings: (Nlayers x Nsamples x Nfeats)

    Ouputs a pdf of the embeddings.
    """

    # setup viz
    n_layers = net.get_n_layers
    z_step = 3.2
    fig = plt.figure()
    fig.set_size_inches(48, 48, forward=True)
    ax = fig.add_subplot(111, projection='3d', proj_type='ortho', computed_zorder=False)
    ax.view_init(elev=35,azim=135)
    ax.dist=8

    vizMapping(net, embeddings, embeddings_nonsequential, Y, grid_us, grid_vs, grid_flat, None, z_step, ax)

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    output_path = './{}.png'.format(net.__class__.__name__)
    plt.savefig(output_path)
    
    # Trim whitespace using ImageMagick convert
    trimmed_path = './{}.png'.format(net.__class__.__name__)
    try:
        # Trim and add a small white border (e.g., 10 pixels)
        subprocess.run([
            'magick',
            output_path,
            '-trim',
            '-bordercolor', 'white',
            '-border', '100x100',
            trimmed_path
        ], check=True)
    except Exception as e:
        print(f'Warning: Could not run convert to trim image: {e}')