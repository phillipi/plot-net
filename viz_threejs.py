import json
import numpy as np

def viz(net, embeddings, embeddings_nonsequential, Y, grid, args):
    export_viz_data(net, embeddings, embeddings_nonsequential, Y, grid, args)

def export_viz_data(net, embeddings, embeddings_nonsequential, Y, grid, args, path='./threejs/viz_data.json'):
    if args.viz_type == 'movie':
        # For movie data, use the first frame to determine n_layers
        n_layers = len(embeddings[0]) - 1
    else:
        # For static data, use the original calculation
        n_layers = len(embeddings) - 1
    
    data = {
        'n_layers': n_layers,
        'embeddings': [e.tolist() for e in embeddings],
        'embeddings_nonsequential': [e.tolist() for e in embeddings_nonsequential],
        'Y': Y.tolist() if Y is not None else [0] * len(embeddings[0]) if len(embeddings) > 0 else [],
        'grid': grid.tolist(),
        'd': net.d,  # for selecting color palette
        'viz_type': args.viz_type
    }
    with open(path, 'w') as f:
        json.dump(data, f)
