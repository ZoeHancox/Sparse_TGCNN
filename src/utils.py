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
from keras.utils import np_utils
import matplotlib.pyplot as plt


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
    
def get_one_hot(df_col):
    encoder = LabelEncoder()
    encoder.fit(df_col)
    encoded_Y = encoder.transform(df_col)
    # convert integers to dummy variables (i.e. one hot encoded)
    dummy_y = np_utils.to_categorical(encoded_Y)
    return dummy_y
    
    
def load_in_y_visit_count(num_codes_to_use):
    initial_label_data = pd.read_csv('../CSV_files/reduced_labels.csv', sep=",", dtype={'dod': 'object',
                                                                   'drug_dose': 'object',
                                                                   'drug_number': 'object',
                                                                   'drug_quantity': 'object',
                                                                   'referral_reason': 'object',
                                                                   'referral_to': 'object',
                                                                   'referral_type': 'object',
                                                                   'referral_urgency': 'object'}) 
    initial_label_data['date'] = pd.to_datetime(initial_label_data.time, format='%Y-%m-%d')
    patient_dates = initial_label_data[['patientnum', 'time']]
    patient_dates_grouped = patient_dates.groupby('patientnum')['time'].apply(set)
    date_set_df = patient_dates.merge(patient_dates_grouped, how='inner', on='patientnum')
    date_set_df = date_set_df.drop_duplicates(subset='patientnum')
    date_set_df['visit_count'] = date_set_df.apply(lambda row: len(row['time_y']), axis=1)
    max_visits = date_set_df['visit_count'].max()
    partitions = pd.cut(date_set_df['visit_count'], bins=[-0.001, 0, 5, max_visits], labels=["Zero", "Low", "High"], retbins=False) 
    partitions.to_frame()
    date_set_df = date_set_df[['patientnum', 'time_y', 'visit_count']]
    row_per_patient = pd.concat([date_set_df, partitions], axis=1)
    row_per_patient.columns = ['patientnum', 'time', 'visit_count', 'visit_util_category']
    
    tensor_input_data = pd.read_pickle("input_data/event_codes_last_100_timesteps_"+str(num_codes_to_use)+"_top_codes.pkl")
    historic_patient_list = tensor_input_data['patientnum'].tolist() # get a list of patients in the historic data
    historic_patient_df = pd.DataFrame (historic_patient_list, columns = ['patientnum'])
    
    df_diff = historic_patient_df[~historic_patient_df['patientnum'].isin(row_per_patient['patientnum'])]
    # add
    df_diff['time']='x'
    df_diff['visit_count'] = 0
    df_diff['visit_util_category'] = 'Zero'
    df_full = pd.concat([row_per_patient, df_diff], ignore_index=True).sort_values(['patientnum'])
    remaining = label_select(df_full, num_codes_to_use)
    
    dummy_y = get_one_hot(remaining.visit_util_category)
    return dummy_y
    
    
def print_label_occurence_order(num_codes_to_use):
    initial_label_data = pd.read_csv('../CSV_files/reduced_labels.csv', sep=",", dtype={'dod': 'object',
                                                                   'drug_dose': 'object',
                                                                   'drug_number': 'object',
                                                                   'drug_quantity': 'object',
                                                                   'referral_reason': 'object',
                                                                   'referral_to': 'object',
                                                                   'referral_type': 'object',
                                                                   'referral_urgency': 'object'}) 
    initial_label_data['date'] = pd.to_datetime(initial_label_data.time, format='%Y-%m-%d')
    patient_dates = initial_label_data[['patientnum', 'time']]
    patient_dates_grouped = patient_dates.groupby('patientnum')['time'].apply(set)
    date_set_df = patient_dates.merge(patient_dates_grouped, how='inner', on='patientnum')
    date_set_df = date_set_df.drop_duplicates(subset='patientnum')
    date_set_df['visit_count'] = date_set_df.apply(lambda row: len(row['time_y']), axis=1)
    max_visits = date_set_df['visit_count'].max()
    partitions = pd.cut(date_set_df['visit_count'], bins=[-0.001, 0, 5, max_visits], labels=["Zero", "Low", "High"], retbins=False) 
    partitions.to_frame()
    date_set_df = date_set_df[['patientnum', 'time_y', 'visit_count']]
    row_per_patient = pd.concat([date_set_df, partitions], axis=1)
    row_per_patient.columns = ['patientnum', 'time', 'visit_count', 'visit_util_category']
    
    tensor_input_data = pd.read_pickle("input_data/event_codes_last_100_timesteps_"+str(num_codes_to_use)+"_top_codes.pkl")
    historic_patient_list = tensor_input_data['patientnum'].tolist() # get a list of patients in the historic data
    historic_patient_df = pd.DataFrame (historic_patient_list, columns = ['patientnum'])
    
    df_diff = historic_patient_df[~historic_patient_df['patientnum'].isin(row_per_patient['patientnum'])]
    # add
    df_diff['time']='x'
    df_diff['visit_count'] = 0
    df_diff['visit_util_category'] = 'Zero'
    df_full = pd.concat([row_per_patient, df_diff], ignore_index=True).sort_values(['patientnum'])
    
    occurence_order = df_full.visit_util_category.unique()
    print("Order of category occurence:", occurence_order)
    return occurence_order
    
    
def label_select(count_and_cat, num_codes_to_use):
    """Select only the patient numbers that appear in the input data
    input: the number of visit count and the category from the label data
    output: labels with reduced"""
    tensor_input_data = pd.read_pickle("input_data/event_codes_last_100_timesteps_"+str(num_codes_to_use)+"_top_codes.pkl")
    patients = tensor_input_data['patientnum'].tolist()
    count_and_cat_remain = count_and_cat[pd.DataFrame(count_and_cat['patientnum'].tolist()).isin(patients).any(1).values]
    return count_and_cat_remain


def logits_to_one_hot(logits, num_classes=3):
    """Args: 
                logits: probabilty of selecting each outcome
                num_classes: number of class outputs to be predicted
       Returns:
               one_hot: one hot array showing what class the model predicts 
    """
    one_hot = np.eye(num_classes)[np.argmax(logits,1)]
    return one_hot
    
    
def average_of_list_of_lists(list_of_lists):
    """Args: 
              list_of_lists: metric for each class for each batch as a list of lists (list)
       Returns:
              ave_of_list_of_lists: average of list of lists to get average metric for each class (array)
    """
    list_of_lists_as_array = np.array(list_of_lists)
    ave_of_list_of_lists = np.average(list_of_lists_as_array, axis=0)
    return ave_of_list_of_lists
    
    

def individual_accuracy_score(y_true, y_pred):
    """Args: 
                y_true: true output
                y_pred: predicted output from the model
       Returns:
               acc_of_each_class: accuracy score for each class (this is the same as recall for categorical though)
    """
    cm = confusion_matrix(np.argmax(y_true,1), np.argmax(y_pred,1))
    acc_of_each_class = cm.astype('float')/cm.sum(axis=1)[:, np.newaxis]
    acc_of_each_class = acc_of_each_class.diagonal()
    return acc_of_each_class
    
    
def normalised_inv_class_proportion(num_in_cat_dict):
    """Args: 
                num_in_cat_dict: the number of patients in each output category (dict)
                :
       Returns:
               low_norm, high_norm, zero_norm: the normalised inverse class proportions for each three classes
    """
    # get the inverse of the proportion of each category
#     low_inv = 1/(num_in_cat_dict['Low']/(num_in_cat_dict['Low']+num_in_cat_dict['High']+num_in_cat_dict['Zero']))
#     high_inv = 1/(num_in_cat_dict['High']/(num_in_cat_dict['Low']+num_in_cat_dict['High']+num_in_cat_dict['Zero']))
#     zero_inv = 1/(num_in_cat_dict['Zero']/(num_in_cat_dict['Low']+num_in_cat_dict['High']+num_in_cat_dict['Zero']))
    N = num_in_cat_dict['Low']+num_in_cat_dict['High']+num_in_cat_dict['Zero']
    low_inv = N/num_in_cat_dict['Low']
    high_inv = N/num_in_cat_dict['High']
    zero_inv = N/num_in_cat_dict['Zero']

    # get the normalised inverse of the proportion
#     low_norm = low_inv / (low_inv + high_inv + zero_inv)
#     high_norm = high_inv / (low_inv + high_inv + zero_inv)
#     zero_norm = zero_inv / (low_inv + high_inv + zero_inv)
    total_inv = low_inv + high_inv + zero_inv
    low_norm = low_inv/total_inv
    high_norm = high_inv/total_inv
    zero_norm = zero_inv/total_inv
    
#     low_norm = (low_inv - zero_inv)/(low_inv - zero_inv)
#     high_norm = (high_inv - zero_inv)/(low_inv - zero_inv)
#     zero_norm = (zero_inv - zero_inv)/(low_inv - zero_inv)
    return low_norm, high_norm, zero_norm



def calc_weighted_loss(class_weights, y_batch, logits):
    weights = tf.compat.v2.reduce_sum(class_weights * y_batch, axis=1)
    unweighted_losses = tf.compat.v2.nn.softmax_cross_entropy_with_logits(y_batch, logits)
    weighted_losses = unweighted_losses * weights
    loss_value = tf.reduce_mean(weighted_losses)
    return loss_value 

    
def list_to_4D_tensor(sparse_list):
    """Converts a list of 4D sparse tensors into a 4D tensor by concatenating.
    sparse_list must contain list of sparse tensors which have an extra dim at the front"""
    sparse_4D_tensor = tf.sparse.concat(axis = 0, sp_inputs = sparse_list)
    return sparse_4D_tensor


def batch_set(indice_set, input_matrices, labels, batchsize=64):
    """indice set can be one of the following:
    train_set_indices
    val_set_indices
    test_set_indices
    
    This function takes a set of randomly selected indices and makes a batch of graphs (lists)"""
    number_of_batches = round(len(indice_set) / batchsize)
    index_partitions = [sorted(indice_set[i::number_of_batches]) for i in range(number_of_batches)]
    #print(len(index_partitions))
    batched_graphs = [[input_matrices[i] for i in index_partition] for index_partition in index_partitions]
    batched_labels = [[labels[i] for i in index_partition] for index_partition in index_partitions]
    return batched_graphs, batched_labels



def load_in_y_event_count():
    """Take in the reduced labels data and create a one-hot encoded label set"""
    event_labels = pd.read_csv('../service_utilisation/clinicalevent_service_util_counts.csv')
    event_labels.columns = ['patientnum', 'count']
    partitions = pd.cut(event_labels['count'], bins=[-0.001, 0, 5, 37], labels=["Zero", "Low", "High"], retbins=False) 
    count_and_cat = pd.concat([event_labels, partitions], axis=1)
    count_and_cat.columns = ['patientnum', 'count', 'util_category']
    count_and_cat_remain = label_select(count_and_cat)
    encoder = LabelEncoder()
    encoder.fit(count_and_cat_remain.util_category)
    encoded_Y = encoder.transform(count_and_cat_remain.util_category)
    # convert integers to dummy variables (i.e. one hot encoded)
    dummy_y = np_utils.to_categorical(encoded_Y)
    return dummy_y
