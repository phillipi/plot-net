import numpy as np
import torch

# create toy data
def mk_dataset():

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
