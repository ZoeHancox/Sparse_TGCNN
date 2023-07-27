import pytest
import os
import tensorflow as tf
from tensorflow import keras
#os.chdir('../')
print(os.getcwd())
from src.TGCNN_layer import TGCNN_layer
from src.whole_model import TGCNN_Model
# TODO: Sort so that the files within src are imported


def test_sparse_convolution():
    """Testing that the TGCNN layer correctly convolves over the 3D 
    tensor. We compare the TGCNN layer to an output calculated by 
    hand in a simple case.

    To get the convolution of the graph using the filter we take the 
    graph 3D matrix and flatten it into a 1D matrix. The filter is 
    also flattened into a 1D matrix and transposed.

    Then we matmul the two vectors and sum the result.
    """

    # Create two graphs
    # Graph 1
    i_list = [(0, 0,0,0), (0, 0,0,1), (0, 1,1,0), (0, 1,1,1)]
    v_list= [1.0,1.0,1.0,1.0]
    g1 = tf.sparse.SparseTensor(i_list, v_list, (1,2,2,2))
    g1 = tf.sparse.reorder(g1)

    # Graph 2
    i_list = [(0, 0,0,0), (0, 0,1,0), (0, 1,0,1), (0, 1,1,1)]
    v_list= [1.0,1.0,1.0,1.0]
    g2 = tf.sparse.SparseTensor(i_list, v_list, (1,2,2,2))
    g2 = tf.sparse.reorder(g2)

    # Batch the two graphs together
    sparse_4D_tensor = tf.sparse.concat(axis = 0, sp_inputs = [g1, g2])

    # Creating the TGCNN_layer model
    # num_nodes, num_time_steps, num_filters, filter_size, stride
    # Constant_filter = True makes a filter with shape=(8,2)
    out = TGCNN_layer(2, 2, 2, 2, 1, exponential_scaling=False, variable_gamma=False, constant_filter=True)

    # Put the graphs into the model
    model_out = out(sparse_4D_tensor)

    # Hand calculated correct answer
    hand_calc_out = tf.constant([[[[18.]], [[-18.]]], [[[18.]], [[-18.]]]])

    #assert model_out==hand_calc_out
    tf.debugging.assert_equal(model_out, hand_calc_out)

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
    