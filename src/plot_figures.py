import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
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
    if run_name is not None:
        plt.savefig("loss_graphs/learning_curve_test_"+run_name+"_"+str(ran_search_num)+".png")
    plt.show()
    

def draw_confusion_mat(y_batch_test, test_logits, class_names, run_name, ran_search_num):
    """
    Draw a confusion matrix using the sklearn.metrics confusion matrix
    package.
    
    Args:
        y_batch_test: one hot encoding arrays to show true class (list of arrays)
        test_logits: array of the probability of each class (np.array)
        class_names: list of class names/ labels (list of strings)
    Returns:
        sns confusion matrix plot.
    """
    y_batch_test_array = np.array(y_batch_test)
    y_true = np.argmax(y_batch_test_array, axis=1)
    y_pred = np.argmax(test_logits, axis=1)
    cn=confusion_matrix(y_true,y_pred)

    sns.heatmap(cn,annot=True, fmt='d', cmap='Blues')

    tick_marks = np.arange(len(class_names)) + 0.5
    plt.yticks(tick_marks, class_names)
    plt.xticks(tick_marks, class_names)
    plt.title("Confusion Matrix for Test Set")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    if run_name is not None:
        plt.savefig("loss_graphs/confusion_mat_"+run_name+"_"+str(ran_search_num)+".png")
    plt.show()


