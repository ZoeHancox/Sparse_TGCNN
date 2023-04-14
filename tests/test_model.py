import pytest
import os
import tensorflow as tf
os.chdir('../')
print(os.getcwd())
from src import TGCNN_layer

def test_1_placeholder():
    """_summary_
    """

    assert  1==1

def test_sparse_graph_reg():
    """Test to check filter deviance function is calculating deviance 
    correctly. This uses a filter from the sparse 3DCNN model with a 
    simple case.
    """

    filt_complex = tf.constant([2.5, 1, 3, 0.5, 1, 3, 0.5, 1, 3, 0.5, 1, 
                                3, 0.5, 1, 3, 0.5, 1, 3, 0.5, 1, 3, 0.5, 
                                1, 3, 0.5, 1, 3])
    filtersize = 3
    deviance = TGCNN_layer.graph_reg.filter_deviance(filter=filt_complex,
                                           filtersize=filtersize)
    exp_deviance = 27.0
    assert deviance == exp_deviance
    