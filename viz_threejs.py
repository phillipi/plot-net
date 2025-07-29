import json
import numpy as np

def viz_static(net, embeddings, embeddings_nonsequential, Y, grid, args):
    export_viz_data(net, embeddings, embeddings_nonsequential, Y, grid)

def export_viz_data(net, embeddings, embeddings_nonsequential, Y, grid, path='./threejs/viz_data.json'):
    data = {
        'n_layers': len(embeddings) - 1,
        'embeddings': [e.tolist() for e in embeddings],
        'embeddings_nonsequential': [e.tolist() for e in embeddings_nonsequential],
        'Y': Y.tolist() if Y is not None else [0] * len(embeddings[0]) if len(embeddings) > 0 else [],
        'grid': grid.tolist(),
        'd': net.d  # for selecting color palette
    }
    with open(path, 'w') as f:
        json.dump(data, f)
