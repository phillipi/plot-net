import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.font_manager as font_manager
from matplotlib.animation import FuncAnimation

def vizMapping(X, Y, grid_us, grid_vs, loss, z_offset, ax):

    # create flattened data for samples in grid
    grid_flat = np.stack((grid_us, grid_vs), axis=2)
    grid_flat = np.reshape(grid_flat, (grid_flat.shape[0]*grid_flat.shape[1], grid_flat.shape[2])) # grid_flat: Nsamples x Nfeats
    grid_flat = torch.tensor(grid_flat.astype(np.float32))

    # run net on the data and on the grid
    output = net(X).detach().numpy() # Nlayers x Nsamples x Nfeats
    output_grid = net.forward_grid(grid_flat).detach().numpy() # Nlayers x Nsamples x Nfeats

    n_layers = output.shape[0]
    n_classes = len(np.unique(Y))

    for l in range(n_layers):

        # plot grid
        ax.plot_wireframe(np.reshape(output_grid[l,:,0],[2,2]), np.reshape(-output_grid[l,:,1],[2,2]), (l*z_step)*np.ones_like(np.reshape(output_grid[l,:,0],[2,2]))+z_offset, rcount=1, ccount=1, color = [0.7,0.7,0.7], linewidth=3.0, linestyle='--', zorder=2*l+0.0001)
        ax.plot_surface(grid_us, grid_vs, (l*z_step)*np.ones_like(grid_us)+z_offset, rcount=1, ccount=1, color = [1.0,1.0,1.0], shade=False, linewidth=0, zorder=2*l+0.001,alpha=0.75)
        ax.plot_wireframe(grid_us, grid_vs, (l*z_step)*np.ones_like(grid_us)+z_offset, rcount=1, ccount=1, color = [0.3,0.3,0.3], linewidth=6.0, zorder=2*l+0.01)
        ax.plot([min_u,max_u],[0,0],[l*z_step+z_offset,l*z_step+z_offset], color=[0.5,0.5,0.5], linewidth=4.0, zorder=2*l)
        ax.plot([0,0],[min_v,max_v],[l*z_step+z_offset,l*z_step+z_offset], color=[0.5,0.5,0.5], linewidth=4.0, zorder=2*l)

        # plot grid mapping
        if l<(n_layers-1):
            for i in range(0,output_grid.shape[1],1):
                ax.plot([grid_flat[i,0], output_grid[l+1,i,0]], [-grid_flat[i,1], -output_grid[l+1,i,1]], [l*z_step+z_offset, (l+1)*z_step+z_offset], '--', marker = 'o', color = [0.7, 0.7, 0.7], linewidth=3, markersize=10, zorder=2*l+1)#, alpha=.1)
                ax.scatter(output_grid[l+1,i,0], -output_grid[l+1,i,1], (l+1)*z_step+z_offset, marker = '>', color = [0.1, 0.1, 0.1], s=80, zorder=2*l+1.01)

            # display layer name
            font_prop = font_manager.FontProperties(size=24)
            ax.text(-z_step, 0, (l+1)*z_step+z_offset, net.get_layer_name(l), fontproperties=font_prop)

        # plot data
        for i in range(output.shape[1]):
          if Y[i]==0:
              color = np.array([255, 0, 0, 255])/255
          else:
              color = np.array([0, 0, 255, 255])/255

          if Y[i]==0:
            if l<n_layers-1:
              ax.plot([output[l,i,0], output[l+1,i,0]], [-output[l,i,1], -output[l+1,i,1]], [l*z_step+z_offset, (l+1)*z_step+z_offset], '--', marker = '.', color=color, linewidth=3, markersize=0, zorder=2*l+1, alpha=.5)
            ax.scatter(output[l,i,0].ravel(), -output[l,i,1].ravel(), (l)*z_step+z_offset, color=color, s=160, alpha=0.5, linewidth=0, zorder=2*l+1.01)
          else:
            if l<n_layers-1:
              ax.plot([output[l,i,0], output[l+1,i,0]], [-output[l,i,1], -output[l+1,i,1]], [l*z_step+z_offset, (l+1)*z_step+z_offset], '--', marker = '.', color=color, linewidth=3, markersize=0, zorder=2*l+1, alpha=.5)
            ax.scatter(output[l,i,0].ravel(), -output[l,i,1].ravel(), (l)*z_step+z_offset, color=color, s=160, alpha=0.5, linewidth=0, zorder=2*l+1.01)

    # display loss
    font_prop = font_manager.FontProperties(size=24)
    ax.text(-z_step/4,0, (n_layers-0.2)*z_step+z_offset, 'Loss: {:.2f}'.format(loss), fontproperties=font_prop)

    ax.set_xlim3d(-2*n_layers/2,2*n_layers/2)
    ax.set_ylim3d(-2*n_layers/2,2*n_layers/2)
    ax.set_zlim3d(-0.5,2*n_layers)
    ax.set_axis_off()
