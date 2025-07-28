# imports
import numpy as np
import torch
import models, train, viz_matplotlib, viz_plotly,viz_threejs, datasets
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot-Net Visualization")
    parser.add_argument('--which_dataset', type=str, default='gaussian_data', help='Dataset function to use from datasets module')
    parser.add_argument('--which_model', type=str, default='linear', help='Model class to use from models module')
    parser.add_argument('--viz_type', type=str, default='static', help='Type of visualization to use (static, movie)')
    parser.add_argument('--d', type=int, default=2, help='Dimensionality of the data')
    parser.add_argument('--N_viz_iter', type=int, default=180, help='Number of frames in the video')
    parser.add_argument('--N_train_iter_per_viz', type=int, default=250, help='Number of training steps per frame')
    parser.add_argument('--train', type=str, default='False', help='Whether to train the net')
    parser.add_argument('--rotate_camera', type=str, default='False', help='Whether to rotate the camera')
    parser.add_argument('--renderer', type=str, default='matplotlib', help='Renderer to use (matplotlib, threejs, plotly)')
    args = parser.parse_args()

    # seed (for replicability)
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)

    # make dataset
    X, Y = datasets.mk_dataset(args.which_dataset, args.d)
    grid = datasets.mk_grid_data(args.d)

    # setup net
    net = models.mk_model(args.which_model, args.d)

    if args.train == 'True':
        # setup optimizer
        lr = 0.002
        optimizer = torch.optim.SGD(net.parameters(), lr=lr)
        criterion = torch.nn.NLLLoss()

        # train the net, storing the embeddings on all layers at each step of optimization
        embeddings, embeddings_nonsequential, losses = train.train(net, X, Y, grid, args.N_viz_iter, args.N_train_iter_per_viz, optimizer, criterion)

        if args.viz_type == 'static':
            embeddings = embeddings[-1]
            embeddings_nonsequential = embeddings_nonsequential[-1]
        
    else:
        # just run inference on the net to get the embeddings
        embeddings = net.forward(X).detach().numpy()
        embeddings_nonsequential = net.forward_nonsequential(grid).detach().numpy()

        if args.viz_type == 'movie':
            # replicate for args.N_viz_iter (this will just give a movie a static net, potentially with camera movement)
            embeddings = np.repeat(embeddings[np.newaxis, :, :], args.N_viz_iter, axis=0)
            embeddings_nonsequential = np.repeat(embeddings_nonsequential[np.newaxis, :, :], args.N_viz_iter, axis=0)
            losses = np.zeros(args.N_viz_iter)

    if args.viz_type == 'movie':
        # visualize the embeddings as a video, showing how they change over steps of optimization
        if args.renderer == 'matplotlib':
            viz_matplotlib.viz_movie(net, embeddings, embeddings_nonsequential, Y, grid, losses, args)
        elif args.renderer == 'threejs':
            raise NotImplementedError("ThreeJS movie visualization not implemented yet")
        elif args.renderer == 'plotly':
            raise NotImplementedError("plotly movie visualization not implemented yet")
    
    elif args.viz_type == 'static':
        # visualize the embeddings as a static plot
        if args.renderer == 'matplotlib':
            viz_matplotlib.viz_static(net, embeddings, embeddings_nonsequential, Y, grid, args)
        elif args.renderer == 'threejs':
            viz_threejs.viz_static(net, embeddings, embeddings_nonsequential, Y, grid, args)
        elif args.renderer == 'plotly':
            viz_plotly.viz_static(net, embeddings, embeddings_nonsequential, Y, grid, args)

    else:
        raise ValueError(f"Visualization type {args.viz_type} not found")
