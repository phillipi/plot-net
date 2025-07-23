# imports
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.font_manager as font_manager
from matplotlib.animation import FuncAnimation

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import time

import utils, models, viz, datasets

# setup directories
layer_embeddings_viz_dir = './'
vid_dir = './'

# seed (for replicability)
seed = 0
np.random.seed(seed)
torch.manual_seed(seed)

# plotting parameters
max_u, max_v = 1, 1
min_u, min_v = -1, -1
z_step = 3.2

# make dataset
X, Y = mk_dataset()

# setup net
net = Net()
net.apply(weights_init) # re-initialize net
criterion = nn.NLLLoss()
lr = 0.002
optimizer = optim.SGD(net.parameters(), lr=lr)
N_viz_iter = 2
N_train_iter_per_viz = 250

output = net(X).detach().numpy() # Nlayers x Nsamples x Nfeats
n_layers = output.shape[0]
z_offset = -z_step*((n_layers-3)/2)

record_type = 'vid' # can't seem to get both video and individual frames to work simultaneously,
                    #  so just select which type of recording you want here

# setup figure
fig = plt.figure()
fig.set_size_inches(48, 48, forward=True)
ax = fig.add_subplot(111, projection='3d', proj_type = 'ortho', computed_zorder=False)
ax.view_init(elev=35,azim=135)
ax.dist=8

# setup grid for visualization
grid_us, grid_vs = np.meshgrid(np.linspace(min_u,max_u,num=2), np.linspace(min_v,max_v,num=2))

# functions for training
def init():
    ax.set_xlim(min_u, max_u)
    ax.set_ylim(min_v, max_v)
    ax.set_zlim(-10, 10)  # Adjust if needed
    return []

def update(ii):
    ax.cla()  # clear current frame
    ax.view_init(elev=35, azim=135)
    ax.dist = 8

    loss = computeLoss(net, criterion, X, Y)
    print(f'loss at frame {ii}: {loss:.4f}')

    vizMapping(X, Y, grid_us, grid_vs, loss, z_offset, ax)

    trainForABit(net, criterion, optimizer, X, Y, N_train_iter_per_viz)
    return []

ani = FuncAnimation(fig, update, frames=N_viz_iter, init_func=init, blit=False)

ani.save(vid_dir + './animation.mp4', fps=60)

!ffmpeg -y -i animation.mp4 -filter:v "crop=iw*0.4:ih*0.6:(iw*0.35):(ih*0.25)" -c:a copy animation_cropped.mp4
