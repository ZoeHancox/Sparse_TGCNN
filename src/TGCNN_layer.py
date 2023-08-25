import numpy as np
import math
#import keras
import pandas as pd
import random
from tensorflow import keras
import tensorflow as tf


class TGCNN_layer(tf.keras.layers.Layer):
    """Applying a filter to a single graph over time slices

        Args:
            input_graphs: 4D tensor of sparse graphs (batch_size, num_nodes, num_nodes, timesteps)
            num_filter: number of graph filters/features to use
            filter_size: number of time steps (t) to use in filters
            parallel_iter (int): experiment with this variable to improve efficiency

        Returns:
            4D output tensor of features (batch_size, num_filters, 1, T-t+1)
    """
    
    def __init__(self, num_nodes, num_time_steps, num_filters, filter_size, stride, variable_gamma=True, 
                 exponential_scaling=True, parallel_iter=100, dtype_weights=tf.float32, no_timestamp=False,
                 conv_test = False, graph_reg_test = False):
        super(TGCNN_layer, self).__init__()
        #self.graphs_4D = list_to_4D_tensor(input_graphs)
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.stride = stride
        
        self.num_nodes = num_nodes
        self.time_steps = num_time_steps # T = the temporal length of the graphs
        self.dtype_weights = dtype_weights
        self.parallel_iter = parallel_iter
        self.variable_gamma = variable_gamma
        self.exponential_scaling = exponential_scaling
        self.no_timestamp = no_timestamp
        
        w_init = tf.random_normal_initializer(stddev = 0.05)#1e-3)

        if conv_test == False and graph_reg_test == False:
            self.w = tf.Variable(
                initial_value=w_init(shape=(self.num_nodes*self.num_nodes*self.filter_size, self.num_filters)),
                trainable=True, dtype=self.dtype_weights, name='3DCNN_Weights')
        elif graph_reg_test == True and conv_test == False:
            filt_complex = tf.constant([2.5, 1, 3, 0.5, 1, 3, 0.5, 1, 3, 0.5, 1, 
                                3, 0.5, 1, 3, 0.5, 1, 3, 0.5, 1, 3, 0.5, 
                                1, 3, 0.5, 1, 3])
            self.w = tf.Variable(initial_value=tf.constant(filt_complex, dtype=self.dtype_weights),
                                 trainable=True, dtype=self.dtype_weights)
        elif conv_test == True and graph_reg_test == False:
            self.w = tf.transpose(tf.constant([[1, 2, 3, 4, 5, 6, 7, 8], [-1, -2, -3, -4, -5, -6,-7, -8]], dtype=self.dtype_weights))
        else:
            print("Can't set two constant tensors for the filter, change either conv_test or graph_reg_test to False :)")

        
        g_init = tf.random_normal_initializer()
        if self.exponential_scaling & self.variable_gamma:
            self.gammat = tf.Variable(initial_value=g_init(shape=(1,1)), trainable=True, dtype=self.dtype_weights,
                                    name='Gammat')
        
               
    def call(self, input_graphs):
        k = tf.constant(1, dtype=tf.int64)
        
        if self.exponential_scaling:
            if self.variable_gamma:
                gamma_max = 10
                gamma_min = 0
                self.gammat = (gamma_max-gamma_min)*tf.nn.sigmoid(self.gammat)+gamma_min
                #print(input_graphs)
                #print(self.gammat)
                #print(-self.gammat*input_graphs)
                input_graphs = tf.sparse.map_values(tf.exp, -self.gammat*input_graphs) # all non-zero elements in sparse tensor (exp(-gamma*x))
            elif self.no_timestamp:
                input_graphs = tf.sparse.map_values(tf.exp, 0)
                
            else:
                input_graphs = tf.sparse.map_values(tf.exp, tf.constant(-1, dtype=tf.float32)*input_graphs)

        
        # Create features by building it up rather than starting with a fill array and replacing numbers
        # First slice matmul over all graphs 
        #print(tf.shape(input_graphs))
        g = tf.sparse.reshape(
                tf.sparse.slice(input_graphs, 
                                [0, 0, 0, 0], 
                                [input_graphs.dense_shape[0],  
                                 self.num_nodes, self.num_nodes, self.filter_size]), [-1,1]) 
        #print(tf.shape(g))
        g = tf.sparse.reshape(g, [input_graphs.dense_shape[0], self.num_nodes*self.num_nodes*self.filter_size])

        g = tf.sparse.sparse_dense_matmul(g, self.w)

        g = tf.expand_dims(g, 2)
        g = tf.expand_dims(g, 2)


        def in_loop(k, g):
            g = tf.concat([g, tf.expand_dims(
                    tf.expand_dims(
                        tf.sparse.sparse_dense_matmul( # must have same dtype
                            tf.sparse.reshape(
                                tf.sparse.reshape(
                                    tf.sparse.slice(input_graphs, 
                                [0, 0, 0, k], 
                                [input_graphs.dense_shape[0],  
                                 self.num_nodes, self.num_nodes, self.filter_size]), [-1,1]), 
                                [input_graphs.dense_shape[0], self.num_nodes*self.num_nodes*self.filter_size]), self.w),2),2)], 3) # concatenate to build output
            return k + self.stride, g
        
        #shape_0 = 
        shape_1 = self.num_filters
        #print(shape_1)
        shape_2 = 1
        #print(shape_2)
        shape_3 = int(((self.time_steps - self.filter_size)/self.stride) +1)
        #print(shape_3)
        _, g = tf.while_loop(lambda k, g: k < self.time_steps-self.filter_size+1, in_loop, [k, g], 
                      shape_invariants=[k.get_shape(), tf.TensorShape([None, None, None, None])], 
                      #shape_invariants=[k.get_shape(), tf.TensorShape([None, shape_1, shape_2, shape_3])],
                              parallel_iterations = self.parallel_iter, swap_memory=False) # k = counter
          
        #print("g (output) shape when stride=", self.stride, ':', g.get_shape())
        return g
    
    
    
    
    def l1_reg(self):
        '''L1 regularization on weights.
        Args:
            w: weights of the filters
        Returns:
            norm: float giving scaled L1 norm of filters.
        '''
        #return tf.reduce_sum(tf.abs(self.w))/tf.size(self.w, out_type=self.dtype_weights)
        return tf.norm(self.w, ord=1)
        
    def l2_reg(self):
        '''L2 regularization on weights.
        Args:
            w: weights of the filters
        Returns:
            norm: float giving scaled L2 norm of filters.
        '''
        #return tf.reduce_sum(tf.square(self.w))/tf.size(self.w, out_type=self.dtype_weights)
        return tf.norm(self.w, ord='euclidean')

    
    


    def graph_reg(self):
        '''Structured L1-regularization to try enforce graph structure. 
            This penalises if there is no feeder event or prior connections.
        Returns:
            deviance: float representing unscaled deviance from perfect graph structure
        '''
        # def filter_deviance(Fi,
        #                     filtersize = self.filter_size):
        #     '''
        #     Args:
        #         Fi: weights [num_nodes, num_nodes, filter_size, num_filters]
        #         filtersize: number of time steps (t) to use in 3D CNN filters
        #     Returns:
        #         deviance: float representing unscaled deviance from perfect graph structure
                
        #     '''
        #     deviance = tf.constant(0.0, dtype=tf.float32)
        #     threshold = tf.constant(1e-3, dtype=tf.float32)
        #     #print(tf.shape(Fi))
        #     Fiabs = tf.abs(Fi)

        #     k = tf.constant(0, dtype=tf.int32)
        #     def in_loop(k, deviance):
        #         print(tf.shape(Fiabs))
        #         where = tf.greater(Fiabs[k], threshold)
        #         cols = tf.equal(tf.reduce_any(where, axis=0), tf.constant(False, dtype=tf.bool))
        #         deviance += tf.reduce_sum(tf.boolean_mask(Fiabs[k+1], cols))
        #         return k + 1, deviance

        #     k, deviance = tf.while_loop(lambda k, deviance: k < filtersize-1, in_loop, [k, deviance])
        #     return deviance
        #print(tf.shape(self.w))
        #print(tf.shape(self.w[:,0]))
        def filter_deviance(self):
            """ Calculates the sum of the absolute values of filter weights that 
            are less than the threshold (threshold) at specific time steps k. 
            This value is used as a measure of deviance from the so-called 'perfect' 
            graph structure. The deviance calculation aims to penalize the absence of 
            feeder events or prior connections in the graph represented by the filter weights.

            Args:
                filter (tf.Tensor): Weights from the graph.
                filtersize (int): Number of time steps (t) to use in 3D CNN filters.

            Returns:
                float: Unscaled deviance from the so-called 'perfect' graph.
            """
            Fiabs = tf.abs(self.w)
            threshold = 1.1
            deviances = []
            k=0
            while k < self.filter_size:
                timestep = Fiabs[k::self.filter_size]
                above_thres = tf.cast(tf.greater(timestep, threshold), tf.bool)
                below_thres = tf.cast(tf.less(timestep, threshold), tf.bool)
                prior_connection = tf.reduce_any(above_thres)
                
                if not prior_connection:
                    deviances.append(tf.reduce_sum(tf.boolean_mask(Fiabs[k+1::self.filter_size], below_thres)))
                
                k += 1

            deviance = tf.reduce_sum(deviances)
            
            return deviance.numpy()
        total_deviance = filter_deviance(self.w[:, 0]) # first filter weights
        for featurenum in range(1, self.w.shape[1]): # loop through the number of filters
            total_deviance += filter_deviance(self.w[:, featurenum])
        return total_deviance/tf.size(self.w, out_type=tf.float32)
    
    
    

