import numpy as np
from src import utils, TGCNN_layer, whole_model, plot_figures
from tensorflow import keras
import tensorflow as tf
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, precision_score, roc_auc_score, recall_score
from csv import writer
from sklearn.preprocessing import OneHotEncoder


def convert_logits_to_probs(logits, y_true):
    """
    Convert logits to probabilities and true predictions.
    """
    logits = tf.squeeze(logits)
    y_true = tf.squeeze(y_true)
    y_true = tf.cast(y_true, tf.float32)
    probs = 1 / (1 + (np.exp(-logits)))
    probs = tf.squeeze(probs)
    # convert probs to binary predictions/binary       
    dummy_pred = np.where(probs > 0.5, 1, 0) 

    return logits, dummy_pred, y_true

def model_metrics(y_true, dummy_pred, logits):
    """
    Gets the accuracy, precision, recall, AUROC, F1 score and calibration slope values for the model.
    """
    acc = accuracy_score(y_true, dummy_pred)

    prec = precision_score(y_true, dummy_pred, average='binary', zero_division=1, pos_label=1)

    recall = recall_score(y_true, dummy_pred, average='binary', zero_division=1, pos_label=1)

    auc = 0
    try:
        auc = roc_auc_score(y_true, dummy_pred, average=None)
    except ValueError:
        pass

    f1 = f1_score(y_true, dummy_pred, average='binary', zero_division=1, pos_label=1)

    cal_slope = utils.calibration_slope(y_true, logits)

    return acc, prec, recall, auc, f1, cal_slope


def train_step(x, y_batch_train, trn_demo_vals, reg_strength, class_weights, model, L1_ablation, L2_ablation, graph_reg_strength, graph_reg_incl, exponential_scaling, weighted_loss, variable_gamma, optimizer, demo):
    x_batch_train = utils.list_to_4D_tensor(x) # takes one batch from the training set

    # Open a GradientTape to record the operations run during the forward pass, which enables auto-differentiation.
    with tf.GradientTape() as tape:

        # Run the forward pass of the layer.
        # The operations that the layer applies to its inputs are going to be recorded on the GradientTape.
        if demo:
            trn_logits = model(x_batch_train, trn_demo_vals, training=True)
        else:
            trn_logits = model(x_batch_train, training=True)

        trn_logits, dummy_pred, y_batch_train = convert_logits_to_probs(trn_logits, y_batch_train)
        
        if weighted_loss:
            trn_loss = utils.calc_weighted_loss(class_weights, y_batch_train, trn_logits)
        else:           
            loss_value_tensor = tf.compat.v2.nn.sigmoid_cross_entropy_with_logits(y_batch_train, trn_logits)
            trn_loss = tf.reduce_mean(loss_value_tensor)

        if L1_ablation:
            trn_loss += reg_strength * model.tg_conv_layer1.l1_reg()

        if L2_ablation:
            trn_loss += reg_strength * model.tg_conv_layer1.l2_reg() # apply L2 reg to the CNN layer

        if graph_reg_incl:
            _, scaled_deviance = model.tg_conv_layer1.graph_reg()
            trn_loss += graph_reg_strength * scaled_deviance


        if variable_gamma and (exponential_scaling==True):
            gamma_val = float(model.tg_conv_layer1.gammat.numpy())
        else:
            gamma_val = 'N/A'


    # Use the gradient tape to automatically retrieve
    # the gradients of the trainable variables with respect to the loss.
    grads = tape.gradient(trn_loss, model.trainable_weights)
    # print("grads:", grads)

    # Run one step of gradient descent by updating
    # the value of the variables to minimize the loss.
    optimizer.apply_gradients(zip(grads, model.trainable_weights))

    #trained_weights = model.get_weights()
    #print("Trained weights:", trained_weights)
    
    acc, prec, recall, auc, f1, cal_slope = model_metrics(y_batch_train, dummy_pred, trn_logits)
    
   
    return trn_logits, trn_loss, acc, prec, recall, auc, f1, cal_slope, model    
    
    
    
    


def val_step(x, y, test_demo_vals, reg_strength,class_weights,model,L1_ablation,L2_ablation,graph_reg_strength, graph_reg_incl, weighted_loss, demo):
    x = utils.list_to_4D_tensor(x) # takes one batch from the validation set
    
    if demo:
        val_logits = model(x, test_demo_vals, training=False)
    else:
        val_logits = model(x, training=False)
    
    val_logits, dummy_pred, y = convert_logits_to_probs(val_logits, y)

    
    if weighted_loss:
        val_loss = utils.calc_weighted_loss(class_weights, y, val_logits, binary=True)
    else:
        loss_value_tensor = tf.compat.v2.nn.sigmoid_cross_entropy_with_logits(y, val_logits)
        val_loss = tf.reduce_mean(loss_value_tensor)

    if L1_ablation:
        val_loss += reg_strength * model.tg_conv_layer1.l1_reg()

    if L2_ablation:
        val_loss += reg_strength * model.tg_conv_layer1.l2_reg() # apply L2 reg to the CNN layer

    if graph_reg_incl:
        _, scaled_deviance = model.tg_conv_layer1.graph_reg()
        val_loss += graph_reg_strength * scaled_deviance
    
    
    acc, prec, recall, auc, f1, cal_slope = model_metrics(y, dummy_pred, val_logits)
    
    
    
   
    return val_logits, val_loss, acc, prec, recall, auc, f1, cal_slope, model
