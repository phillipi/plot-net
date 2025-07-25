import numpy as np

def train_for_a_bit(net, optimizer, criterion, X, Y, N_train_iter_per_viz):
    for iter in range(N_train_iter_per_viz):
        optimizer.zero_grad()
        loss = net.loss(X,Y,criterion)
        loss.backward()
        optimizer.step()

    return loss

def train(net, X, Y, grid_flat, N_viz_iter, N_train_iter_per_viz, optimizer, criterion):

    embeddings = []
    embeddings_nonsequential = []
    losses = []

    for iter in range(N_viz_iter):
        losses.append(train_for_a_bit(net, optimizer, criterion, X, Y, N_train_iter_per_viz).detach().numpy())
        embeddings.append(net.forward(X).detach().numpy())
        embeddings_nonsequential.append(net.forward_nonsequential(grid_flat).detach().numpy())
        print(f'loss at frame {iter}: {losses[-1]:.4f}')

    embeddings = np.stack(embeddings) # Nsteps x Nlayers x Nsamples x Nfeats
    embeddings_nonsequential = np.stack(embeddings_nonsequential) # Nsteps x Nlayers x Nsamples x Nfeats
    losses = np.stack(losses) # Nsteps x 1

    return embeddings, embeddings_nonsequential, losses
