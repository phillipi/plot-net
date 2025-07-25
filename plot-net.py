# imports
import numpy as np
import torch
import models, train, viz, datasets
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot-Net Visualization")
    parser.add_argument('--which_dataset', type=str, default='gaussian_data', help='Dataset function to use from datasets module')
    parser.add_argument('--which_model', type=str, default='linear', help='Model class to use from models module')
    parser.add_argument('--viz_type', type=str, default='static', help='Type of visualization to use')
    parser.add_argument('--movie_type', type=str, default='mp4', help='Type of movie to save')
    parser.add_argument('--d', type=int, default=2, help='Dimensionality of the data')
    args = parser.parse_args()

    # seed (for replicability)
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)

    # make dataset
    X, Y = datasets.mk_dataset(args.which_dataset, args.d)
    if args.d == 2:
        grid_us, grid_vs, grid_flat = datasets.mk_grid_data(args.d)
        grid_ws = None
    elif args.d == 3:
        grid_us, grid_vs, grid_ws, grid_flat = datasets.mk_grid_data(args.d)
    else:
        raise ValueError('Invalid dimensionality: {}'.format(args.d))

    # setup net
    net = models.mk_model(args.which_model, args.d)

    if args.viz_type == 'training_movie':

        # setup optimizer
        lr = 0.002
        optimizer = torch.optim.SGD(net.parameters(), lr=lr)
        criterion = torch.nn.NLLLoss()

        # setup viz params
        N_viz_iter = 180 # number of frames in the video
        N_train_iter_per_viz = 250 # number of training steps per frame

        # train the net, storing the embeddings on all layers at each step of optimization
        embeddings, embeddings_nonsequential, losses = train.train(net, X, Y, grid_flat, N_viz_iter, N_train_iter_per_viz, optimizer, criterion)

        # now visualize the embeddings as a video, showing how they change over steps of optimization
        viz.viz_movie(net, embeddings, embeddings_nonsequential, Y, grid_us, grid_vs, grid_ws, grid_flat, losses, args)
    
    elif args.viz_type == 'static':
        # get the embeddings of the net
        embeddings = net.forward(X).detach().numpy()
        embeddings_nonsequential = net.forward_nonsequential(grid_flat).detach().numpy()

        # now visualize the embeddings as a static plot
        viz.viz_static(net, embeddings, embeddings_nonsequential, Y, grid_us, grid_vs, grid_ws, grid_flat, args)

    else:
        raise ValueError(f"Visualization type {args.viz_type} not found")
