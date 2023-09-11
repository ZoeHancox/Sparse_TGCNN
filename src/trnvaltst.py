import numpy as np
import math
#import keras
import time
import pandas as pd
import random
from src import utils, TGCNN_layer, whole_model
from early_stopping import EarlyStopping
from tensorflow import keras
import tensorflow as tf
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, precision_score, roc_auc_score, recall_score
from csv import writer
from sklearn.preprocessing import OneHotEncoder


def train_step(x,y_batch_train,reg_strength,class_weights,model,L1_ablation, L2_ablation, graph_reg_strength, graph_reg_incl, weighted_loss, variable_gamma, optimizer):
    x_batch_train = utils.list_to_4D_tensor(x) # takes one batch from the training set
    #initial_weights = model.get_weights()
    #print("Initial weights:", initial_weights)
    #print(y_batch_train)
    #print(x_batch_train)
    #print(tf.shape(x_batch_train))
    # Open a GradientTape to record the operations run during the forward pass, which enables auto-differentiation.
    with tf.GradientTape() as tape:

        # Run the forward pass of the layer.
        # The operations that the layer applies to its inputs are going to be recorded on the GradientTape.
        
        trn_logits = model(x_batch_train, training=True)  # Logits (probability of output being 1)
        dummy_pred = utils.logits_to_one_hot(trn_logits)
        print("trn_logits:", trn_logits)
        print("dummy_pred", dummy_pred)
        # print("Type of dummy_pred is:")
        # print(type(dummy_pred))
        # print("Type of y_batch_train is:")
        # print(type(y_batch_train))

        # print("Values are:")
        # print("dummy_pred")
        # print(dummy_pred)
        # print("y_batch_train")
        # print(y_batch_train)

        # print("Shapes:")
        # print("dummy_pred")
        # print(dummy_pred.shape)
        # print("y_batch_train")
        # print(y_batch_train.shape)

        encoder = OneHotEncoder(sparse=False)
        y_batch_train = encoder.fit_transform(y_batch_train)

        # print("New Shapes:")
        # print("dummy_pred")
        # print(dummy_pred.shape)
        # print("y_batch_train")
        # print(y_batch_train.shape)

        if weighted_loss:
            trn_loss = utils.calc_weighted_loss(class_weights, y_batch_train, trn_logits)
            #print(loss_value)
        else:
            loss_value_tensor = tf.compat.v2.nn.softmax_cross_entropy_with_logits(y_batch_train, trn_logits)
            #print(loss_value_tensor)
            trn_loss = tf.reduce_mean(loss_value_tensor)
            #print(trn_loss)
            #trn_loss = cce_loss_fn(y_batch_train, trn_logits)
#                 break

        if L1_ablation:
            #loss_value += graph_reg_strength * graphconv1.graph_reg() + l1_reg_strength * graphconv1.l1_reg()
            trn_loss += reg_strength * model.tg_conv_layer1.l1_reg()

        if L2_ablation:
            trn_loss += reg_strength * model.tg_conv_layer1.l2_reg() # apply L2 reg to the CNN layer

        if graph_reg_incl:
            _, scaled_deviance = model.tg_conv_layer1.graph_reg()
            trn_loss += graph_reg_strength * scaled_deviance #model.tg_conv_layer1.graph_reg()

            #loss += graph_reg_strength * (graphconv1.graph_reg() + graphconv2.graph_reg()) + l1_reg_strength * (graphconv1.l1_reg() + graphconv2.l1_reg())

        if variable_gamma:
            gamma_val = float(model.tg_conv_layer1.gammat.numpy()) # gamma doesn't seem to be training correctly atm
        else:
            gamma_val = 'N/A'


    #print("Trainable weights:", model.trainable_weights)
    # Use the gradient tape to automatically retrieve
    # the gradients of the trainable variables with respect to the loss.
    grads = tape.gradient(trn_loss, model.trainable_weights)
    # print("grads:", grads)

    # Run one step of gradient descent by updating
    # the value of the variables to minimize the loss.
    optimizer.apply_gradients(zip(grads, model.trainable_weights))

    #trained_weights = model.get_weights()
    #print("Trained weights:", trained_weights)
    
    trn_acc_metric = accuracy_score(y_batch_train, dummy_pred)
    
    trn_prec_metric = precision_score(y_batch_train, dummy_pred, average=None, zero_division=1)
    indiv_trn_prec_metric = precision_score(y_batch_train, dummy_pred, average='weighted', zero_division=1)
    
    trn_recall_metric = recall_score(y_batch_train, dummy_pred, average=None, zero_division=1)
    indiv_trn_recall_metric = recall_score(y_batch_train, dummy_pred, average='weighted', zero_division=1)


    # trn_auc_metric = roc_auc_score(y_batch_train, dummy_pred, average=None, multi_class='ovo')
    trn_auc_metric = 0
    try:
        trn_auc_metric = roc_auc_score(y_batch_train, dummy_pred, average=None, multi_class='ovo')
    except ValueError:
        pass

    
    # indiv_trn_auc_metric = roc_auc_score(y_batch_train, dummy_pred, average='weighted', multi_class='ovo')
    indiv_trn_auc_metric = 0
    try:
        indiv_trn_auc_metric = roc_auc_score(y_batch_train, dummy_pred, average='weighted', multi_class='ovo')
    except ValueError:
        pass

    trn_f1_metric = f1_score(y_batch_train, dummy_pred, average=None, zero_division=1)
    indiv_trn_f1_metric = f1_score(y_batch_train, dummy_pred, average='weighted', zero_division=1)

    
    #print("Actual y", y_batch_train)
    #print("Predicted y", dummy_pred)    
   
    return trn_logits, trn_loss, trn_acc_metric, trn_prec_metric, trn_recall_metric, trn_auc_metric, trn_f1_metric, indiv_trn_prec_metric, indiv_trn_recall_metric, indiv_trn_auc_metric, indiv_trn_f1_metric    
    
    
    
    
    
    
    
    




def val_step(x,y,reg_strength,class_weights,model,L1_ablation,weighted_loss):
    x = utils.list_to_4D_tensor(x) # takes one batch from the validation set

    val_logits = model(x, training=False)
    if L1_ablation:
        
        if weighted_loss:
            val_loss = utils.calc_weighted_loss(class_weights, y, val_logits)
            val_loss += reg_strength * model.tg_conv_layer1.l1_reg()
        else:
            val_loss = tf.compat.v2.nn.softmax_cross_entropy_with_logits(y, val_logits)
            val_loss = tf.reduce_mean(val_loss)
            val_loss += reg_strength * model.tg_conv_layer1.l1_reg()
    else:

        val_loss = utils.calc_weighted_loss(class_weights, y, val_logits)
    
        if weighted_loss:
            val_loss = utils.calc_weighted_loss(class_weights, y, val_logits)
        else:
            val_loss = tf.compat.v2.nn.softmax_cross_entropy_with_logits(y, val_logits)
            val_loss = tf.reduce_mean(val_loss)

    
    dummy_pred = utils.logits_to_one_hot(val_logits)

    y_new=y
    y=utils.logits_to_one_hot(y_new)
    
    val_acc_metric = accuracy_score(y, dummy_pred)
    #acc_indiv_score = individual_accuracy_score(y, dummy_pred) # same as recall
    #train_all_classes_acc_list.append(acc_indiv_score)
    
    val_prec_metric = precision_score(y, dummy_pred, average='weighted', zero_division=1)
    indiv_val_prec_metric = precision_score(y, dummy_pred, average=None, zero_division=1)
    
    val_recall_metric = recall_score(y, dummy_pred, average='weighted', zero_division=1)
    indiv_val_recall_metric = recall_score(y, dummy_pred, average=None, zero_division=1)
    
    # val_auc_metric = roc_auc_score(y, dummy_pred, average='weighted', multi_class='ovo')
    val_auc_metric = 0
    try:
        val_auc_metric = roc_auc_score(y, dummy_pred, average='weighted', multi_class='ovo')
    except ValueError:
        pass

    # indiv_val_auc_metric = roc_auc_score(y, dummy_pred, average=None, multi_class='ovo')
    indiv_val_auc_metric = 0
    try:
        indiv_val_auc_metric = roc_auc_score(y, dummy_pred, average=None, multi_class='ovo')
    except ValueError:
        pass
    
    val_f1_metric = f1_score(y, dummy_pred, average='weighted', zero_division=1)
    indiv_val_f1_metric = f1_score(y, dummy_pred, average=None, zero_division=1)
    
    #print("validation loss test=", val_loss)
    
   
    return val_logits, val_loss, val_acc_metric, val_prec_metric, val_recall_metric, val_auc_metric, val_f1_metric, indiv_val_prec_metric, indiv_val_recall_metric, indiv_val_auc_metric, indiv_val_f1_metric
