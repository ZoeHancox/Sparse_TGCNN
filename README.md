# Sparse TG-CNN
## University of Leeds

### About the Repository

This repository contains the code for a sparse implementation for Temporal Graph-based 3D Convolutional Neural Networks (TG-CNNs).

In this research we represent graphs as 3D tensors, where the cells contain the elapsed time between two event codes being recorded:

![gif of 3D tensor construction representing the temporal graph](documentation/TG-CNN_build.gif)

A link to the original paper containing the non-sparse implementation for massive online open course data to predict student dropout can be found [here](https://link.springer.com/chapter/10.1007/978-3-031-16564-1_34).

_**Note:** Only public or fake data are shared in this repository._

### Project Stucture

The main code is found in the `src` folder of the repository (see Usage below for more information)

```
.
├── documentation                       # Background, explanations
├── src                                 # Source files
├────> create_fake_patients.py          # Generate fake patient data to use in model
├────> TGCNN_layer.py                   # Sparse 3D CNN layer 
├────> trnvaltst.py                     # Train, validate and test functions
├────> utils.py                         # Helpful functions
├────> whole_model.py                   # Model with all layers
├── tests                               # Test using pytest
├── .gitignore
├── README.md
└── requirements.txt
└── TGCNN_test_fake_patients.ipynb


```
### Built With 

[![Python v3.9](https://img.shields.io/badge/python-v3.9-blue.svg)](https://www.python.org/downloads/release/python-390/)

- [Tensorflow](https://www.tensorflow.org/)
- [Numpy](https://numpy.org/)
- [Pandas](https://pandas.pydata.org/)

### Getting Started

#### Installation

To clone the repo:

`git clone https://github.com/ZoeHancox/Sparse_TGCNN.git`

To create a suitable environment we suggest:
- Build conda environment via `conda create --name tgcnn_env python=3.9`
- Activate environment `conda activate tgcnn_env`
- Install requirements via `python -m pip install -r ./requirements.txt`

### Usage

Run through the `TGCNN_test_fake_patients.ipynb` notebook to find out how to use this code.

### Testing

To test some of the functions you can run the following from the top directory:
`pytest tests/test_model.py`.
