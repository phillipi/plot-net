import numpy as np
import torch

def mk_binary_classification(d=2):

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

    if d == 3:
        z_noise = np.random.normal(0, np.sqrt(0.02), size=(X.shape[0], 1))
        X = np.concatenate([X, z_noise], axis=1)

    X = torch.tensor(X.astype(np.float32))
    Y = torch.tensor(Y.astype(np.int_))

    return X, Y

def mk_ternary_classification(d=3):
    if d != 3:
        raise ValueError('Ternary classification only supported in 3D')
    Nsamples = 25
    # Means for three classes
    means = [
        [-0.7, -0.7, -0.7],
        [0.7, 0.0, -0.7],
        [0.0, 0.35, 0.7]
    ]
    # Different diagonal covariances
    covs = [
        np.diag([0.05, 0.08, 0.05]),
        np.diag([0.08, 0.05, 0.05]),
        np.diag([0.05, 0.05, 0.08])
    ]
    Xs = []
    Ys = []
    for i, (mean, cov) in enumerate(zip(means, covs)):
        Xs.append(np.random.multivariate_normal(mean, cov, Nsamples))
        Ys.append(np.full(Nsamples, i))
    X = np.concatenate(Xs, axis=0)
    Y = np.concatenate(Ys, axis=0)
    X = torch.tensor(X.astype(np.float32))
    Y = torch.tensor(Y.astype(np.int_))
    return X, Y

def mk_gaussian_data(d=2):
    Nsamples = 50

    # d-dimensional mean and custom diagonal covariance
    mean = np.zeros(d)
    # Covariance: 0.1, 0.5, 0.25, 0.125, ...
    cov_diag = [0.1 for _ in range(d)] # [0.1] + [0.05**i for i in range(1, d)]
    cov = np.diag(cov_diag)
    X = np.random.multivariate_normal(mean, cov, Nsamples)
    Y = None

    X = torch.tensor(X.astype(np.float32))

    return X, Y

# create data
def mk_dataset(which_dataset, d):

    if which_dataset == 'binary_classification':
        return mk_binary_classification(d)
    elif which_dataset == 'ternary_classification':
        return mk_ternary_classification(d)
    elif which_dataset == 'gaussian_data':
        return mk_gaussian_data(d)
    else:
        raise ValueError(f"Dataset {which_dataset} not found")

# create data on a grid
def mk_grid_data(d=2):

    # setup grid of inputs
    min_val, max_val = -1, 1
    grid_axes = [np.linspace(min_val, max_val, num=2) for _ in range(d)]
    mesh = np.meshgrid(*grid_axes, indexing='ij'
    )
    # create flattened data for samples in grid
    grid_flat = np.stack(mesh, axis=-1)
    grid_flat = np.reshape(grid_flat, (-1, d)) # grid_flat: Nsamples x Nfeats
    grid_flat = torch.tensor(grid_flat.astype(np.float32))

    return (*mesh, grid_flat)