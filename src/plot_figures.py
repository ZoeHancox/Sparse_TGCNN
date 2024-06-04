import numpy as np
import tensorflow as tf
import math
import keras
import time
from datetime import timedelta
import pandas as pd
import random
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn.calibration import calibration_curve
from keras.utils import np_utils
import matplotlib.pyplot as plt
import seaborn as sns


def plot_loss_curve(train_loss, val_loss, test_lost, run_name, ran_search_num):
    font_size = 20
    fig, ax1 = plt.subplots(figsize=(8,6))
    a, = ax1.semilogy(range(1,len(train_loss)+1),train_loss,  label='Training', alpha=0.8, linewidth=2)
    b, = ax1.semilogy(range(1,len(val_loss)+1),val_loss,  label='Validation', alpha=0.8, linewidth=2)
    c, = ax1.semilogy(range(1,len(test_lost)+1),test_lost,  label='Test', alpha=0.8, linewidth=2)

    # find position of lowest validation loss
    minposs = val_loss.index(min(val_loss))+1 
    early_stop_line = plt.axvline(minposs, linestyle='--', color='r',label='Early Stopping Checkpoint', linewidth=2)

    plt.xlabel('Epochs', fontsize=font_size)
    plt.ylabel('Loss', fontsize=font_size)
    #plt.ylim(0, 1.0) # consistent scale
    plt.xlim(1, len(train_loss)+1) # consistent scale


    p = [a, b, c]
    ax1.legend([a,b,c, early_stop_line], ['Training', 'Validation', 'Testing', 
                                          'Early Stopping Checkpoint'], loc= 'upper center', fontsize=font_size)


    plt.grid(False)
    plt.tight_layout()
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.savefig("loss_graphs/learning_curve_test_"+run_name+"_"+str(ran_search_num)+".png")
    plt.show()
    

def draw_confusion_mat(y_batch_test, test_logits, class_names, run_name, ran_search_num, data_type="T"):
    """
    Draw a confusion matrix using the sklearn.metrics confusion matrix
    package.
    
    Args:
        y_batch_test: one hot encoding arrays to show true class (list of arrays)
        test_logits: array of the probability of each class (np.array)
        class_names: list of class names/ labels (list of strings)
        data_type: name of the data used (printed in the confusion matrix title)
    Returns:
        sns confusion matrix plot.
    """
    y_batch_test_array = np.array(y_batch_test)
    y_true = np.squeeze(y_batch_test_array)
    
    # convert logits to probabilities and then to binary classifications
    odds = np.exp(test_logits)
    np.seterr(invalid='ignore')
    probs = odds / (1 + odds)
    y_pred = np.where(probs > 0.5, 1, 0) 
    
    cn=confusion_matrix(y_true, y_pred)

    sns.heatmap(cn,annot=True, fmt='d', cmap='Blues')

#     tick_marks = np.arange(len(class_names)) + 0.5
#     plt.yticks(tick_marks, class_names)
#     plt.xticks(tick_marks, class_names)
    
    if data_type == "T":
        input_data_group = "Unbalanced Holdout"
    else:
        input_data_group = "Validation Set for CV (final batch)"
        
    plt.title("Confusion Matrix for: "+input_data_group+" Data")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    if run_name is not None:
        plt.savefig("confusion_mat/"+run_name+"test_"+str(ran_search_num)+".png")
    plt.show()
    
    
    
    
def prob_histo(probabilities, true_y):
    fig, ax1 = plt.subplots(figsize=(8, 6))

    # Plot the histogram of probabilities on the first y-axis
    sns.histplot(probabilities, bins=10, kde=True, color='#4CAF50', edgecolor='black', alpha=0.7, label='Probability', ax=ax1)

    # Add title and labels to the first y-axis
    ax1.set_title('Distribution of Probabilities and True Labels', fontsize=16)
    ax1.set_xlabel('Probability', fontsize=12)
    ax1.set_ylabel('Frequency (Probability)', fontsize=12)

    # Add grid for better readability to the first y-axis
    ax1.grid(axis='y', linestyle='--', alpha=0.7)

    # Customize tick labels for the first y-axis
    ax1.tick_params(axis='both', labelsize=10)

    ax1.set_xlim(0, 1)
    # Create a second y-axis to overlay the countplot
    ax2 = ax1.twinx()

    # Plot the countplot of true_y on the second y-axis
    #true_y = np.concatenate(true_y)
    #true_y_df = pd.DataFrame({'true': true_y})
    sns.histplot(true_y, bins=2, kde=False, color='blue', alpha=0.3, label='True', ax=ax2)
    #sns.countplot(x='true', data=true_y_df, color='blue', alpha=0.4, label='True', ax=ax2)
    

    # Add labels to the second y-axis
    ax2.set_ylabel('Frequency (True Label)', fontsize=12)

    # Customize tick labels for the second y-axis
    ax2.tick_params(axis='y', labelsize=10)
    
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    
    # Show the combined plot
    plt.show()

def draw_calibration_curve(true_y, pred_y, run_name, ran_search_num):
    """
    Plot a calibration curve for a binary model
    
    Args:
        true_y (list): list of single integer arrays with 0 or 1 depending on the class
        pred_y (logits tensor): logits probability of each class shape=(number_of_timestamps, 2)
    Returns:
        Matplotlib figure: Calibration curve showing mean predicted value vs fraction of positives
    """
       
    probs = 1 / (1 + (np.exp(-pred_y)))
    prob_histo(probs, true_y)
    
    fraction_of_pos, mean_pred_value = calibration_curve(true_y, probs, n_bins=10)

    plt.plot(mean_pred_value, fraction_of_pos, '-', label='Overall Calibration')

    plt.plot([0, 1], [0, 1], '--', label='Ideal Calibration')
    plt.xlabel('Mean Predicted Value')
    plt.ylabel('Fraction of Positives')
    plt.legend()
    if run_name is not None:
        plt.savefig("calibration_curves/"+run_name+"test_"+str(ran_search_num)+".png")
    plt.show()



