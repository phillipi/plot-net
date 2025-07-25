# plot-net
Tools for visualizing neural nets

## setup (on mac/linux)
```
python3 -m venv venv
source venv/bin/activate
pip install torch matplotlib numpy
```

## run a model on some data and visualize the embeddings as an image:

`python plot-net.py --which_dataset <dataset_name> --which_model <model_name> --viz_type static`

Example:

`python plot-net.py --which_dataset gaussian_data --which_model linear --viz_type static`

Output:

![Image of linear layer should appear here](img/LinearLayer.png)



## train a model on some data and visualize the evolution of the embeddings over iters as a movie

`python plot-net.py --which_dataset <dataset_name> --which_model <model_name> --viz_type training_movie`

Example:

`python plot-net.py --which_dataset binary_classification --which_model MySimpleNet --viz_type training_movie`

Output:

![Gif of training a net should appear here](animations/MySimpleNet.gif)
