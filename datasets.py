import numpy as np
import torch

def mk_binary_classification():
    Nsamples = 50

    # center circle
    mean = (0,0)
    cov = [[0.02,0],[0,0.02]]
    X = np.random.multivariate_normal(mean,cov,(Nsamples))
    Y = np.zeros(Nsamples)

    # surrounding moon
    xx = np.linspace(-np.pi/2,np.pi-np.pi/2,Nsamples)#np.linspace(-np.pi/4,np.pi-np.pi/4,Nsamples)
    X2 = 0.75*np.stack([np.sin(xx), np.cos(xx)], axis=1)
    mean = (0,0)
    cov = [[0.02,0],[0,0.02]]
    X2 += np.random.multivariate_normal(mean,cov,(Nsamples))
    Y2 = np.ones(Nsamples)

    X = np.concatenate([X,X2], axis=0)
    Y = np.concatenate([Y,Y2], axis=0)

    X = torch.tensor(X.astype(np.float32))
    Y = torch.tensor(Y.astype(np.int_))

    return X, Y

def mk_gaussian_data():
    Nsamples = 50

    # center circle
    mean = (0,0)
    cov = [[0.05,0],[0,0.1]]
    X = np.random.multivariate_normal(mean,cov,(Nsamples))
    Y = None

    X = torch.tensor(X.astype(np.float32))

    return X, Y

# create data
def mk_dataset(which_dataset):

    if which_dataset == 'binary_classification':
        return mk_binary_classification()
    elif which_dataset == 'gaussian_data':
        return mk_gaussian_data()
    else:
        raise ValueError(f"Dataset {which_dataset} not found")

# create data on a grid
def mk_grid_data():

    # setup grid of inputs
    min_u, max_u, min_v, max_v = -1, 1, -1, 1 
    grid_us, grid_vs = np.meshgrid(np.linspace(min_u,max_u,num=2), np.linspace(min_v,max_v,num=2))

    # create flattened data for samples in grid
    grid_flat = np.stack((grid_us, grid_vs), axis=2)
    grid_flat = np.reshape(grid_flat, (grid_flat.shape[0]*grid_flat.shape[1], grid_flat.shape[2])) # grid_flat: Nsamples x Nfeats
    grid_flat = torch.tensor(grid_flat.astype(np.float32))

    return grid_us, grid_vs, grid_flat