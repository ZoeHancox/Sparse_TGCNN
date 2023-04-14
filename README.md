# Sparse TG-CNN
## University of Leeds

### About the Repository

This repository contains the code for a sparse implementation for Temporal Graph-based 3D Convolutional Neural Networks (TG-CNNs). 

A link to the original paper containing the non-sparse implementation for massive online open course data to predict student dropout can be found [here](https://link.springer.com/chapter/10.1007/978-3-031-16564-1_34).

_**Note:** Only public or fake data are shared in this repository._

### Project Stucture

The main code is found in the `src` folder of the repository (see Usage below for more information)

```
.
├── src                                 # Source files
├────> create_fake_patients.py           # Generate fake patient data to use in model
├────> TGCNN_layer.py                    # Sparse 3D CNN layer 
├────> trnvaltst.py                      # Train, validate and test functions
├────> utils.py                          # Helpful functions
├────> whole_model.py                    # Model with all layers
├── tests                               # Test using pytest
├── .gitignore
├── README.md
└── requirements.txt
└── TGCNN_test_fake_patients.ipynb
```

### Getting Started

To clone the repo:

`git clone https://github.com/ZoeHancox/Sparse_TGCNN.git`

To create a suitable environment we suggest:
- Build conda environment via `conda create --name tgcnn_env python=3.9`
- Activate environment `conda activate tgcnn_env`
- Install requirements via `python -m pip install -r ./requirements.txt`

### Testing

To test some of the functions you can run the following from the top directory:
`pytest tests/test_model.py`.
