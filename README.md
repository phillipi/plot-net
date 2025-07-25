# plot-net
Tools for visualizing neural nets

## setup (on mac/linux)
```
git clone git@github.com:phillipi/plot-net.git
cd plot-net

python3 -m venv venv
source venv/bin/activate
pip install torch matplotlib numpy
```

## run a model on some data and visualize the embeddings as an image:

`python plot-net.py --which_dataset <dataset_name> --which_model <model_name> --viz_type static`

Example:

`python plot-net.py --which_dataset gaussian_data --which_model linear --viz_type static`

Output:

<div align="center">
  <img src="img/LinearLayer.png" alt="Image of a linear layer should appear here" width="350"/>
</div>



## train a model on some data and visualize the evolution of the embeddings over iters as a movie

`python plot-net.py --which_dataset <dataset_name> --which_model <model_name> --viz_type training_movie`

Example:

`python plot-net.py --which_dataset binary_classification --which_model MySimpleNet --viz_type training_movie`

Output (click to play the video):

<div align="center">
  <a href="https://raw.githubusercontent.com/phillipi/plot-net/main/animations/MySimpleNet.mp4"><img src="img/MySimpleNet.png" alt="Link to video should appear here" width="400"/></a>
</div>
