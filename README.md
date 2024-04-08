# Sparse TG-CNN

### About the Repository

This repository contains the code for a sparse implementation for Temporal Graph-based 3D Convolutional Neural Networks (TG-CNNs).

In this research we represent graphs as 3D tensors, where the cells contain the elapsed time between two event codes being recorded:

![gif of 3D tensor construction representing the temporal graph](documentation/TG-CNN_build.gif)

A link to the original paper containing the non-sparse implementation for massive online open course data to predict student dropout can be found [here](https://eprints.whiterose.ac.uk/205293/1/TG-CNNs_for_Online_Course_Dropout_Prediction_03.pdf).

_**Note:** Only fake data are shared in this repository. We randomly choose values to fill our fictitious dataframes. For this reason when this code is run the model will not train well. The data we use for this project is secure ResearchOne data, therefore we can't share it with the public._

### Project Stucture

The main code is found in the `src` folder of the repository (see Usage below for more information)

```
.
├── documentation                       # Background, explanations
├── src                                 # Source files
├────> create_fake_patients.py          # Generate fake patient data to use in model
├────> TGCNN_layer.py                   # Sparse 3D CNN layer 
├────> trnvaltst_sigmoid_oned.py        # Train, validate and test functions for multilabel outcomes
├────> trnvaltst_sigmoid_oned.py        # Train, validate and test functions for binary outcomes
├────> utils.py                         # Helpful functions
├────> plot_figures.py                  # Some more helpful functions for plotting figures
├────> whole_model.py                   # Model with all layers and no demographic input
├────> whole_model_demographics.py      # Model with all layers and demographic input
├── tests                               # Test using pytest
├── .gitignore
├── README.md
├── early_stopping_cv.py                # Checkpointing and stopping the model before it overfits
└── requirements.txt
└── TGCNN_test_fake_patients.ipynb      # Notebook where the main code is ran


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

To run the baseline models to compare to the `TGCNN` model, run through the notebooks in `baseline_models/`.

To get the test results pre-calibration and get the logits prepared for recalibration run `results/generate_scores_and_recal_prep.ipynb` and change the `run_name` to the run name you assigned to your model.

To recalibrate the model on Test 1, and test the recalibrated model on Test 2 run the **R** code `recalibration/recalibration_code.R`. Enter the run name of the model. You can choose to do subgroup analysis on your model by entering `y` when prompted by 'Do you want to perform subgroup analysis with this model?'.

### Testing

To test some of the functions you can run the following from the top directory:
`pytest tests/test_model.py`.
