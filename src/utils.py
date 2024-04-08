import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from keras.utils import np_utils
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.calibration import calibration_curve
import warnings 
warnings.simplefilter('ignore', np.RankWarning)
warnings.filterwarnings('ignore')


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
    #plt.ylim(0, 1.0)
    plt.xlim(1, len(train_loss)+1)


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
    
      
    
def print_label_occurence_order(num_codes_to_use):
    initial_label_data = pd.read_csv('input_data/reduced_labels.csv', sep=",", dtype={'dod': 'object',
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


# def logits_to_one_hot(logits, num_classes=3):
#     """Args: 
#                 logits: probabilty of selecting each outcome
#                 num_classes: number of class outputs to be predicted
#        Returns:
#                one_hot: one hot array showing what class the model predicts 
#     """
#     one_hot = np.eye(num_classes)[np.argmax(logits,1)]
#     return one_hot
    
    
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
    
    
def normalised_inv_class_proportion(num_in_cat_dict, class1='Low', class2='High', class3='Zero'):
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
    N = num_in_cat_dict[class1]+num_in_cat_dict[class2]+num_in_cat_dict[class3]
    low_inv = N/num_in_cat_dict[class1]
    high_inv = N/num_in_cat_dict[class2]
    zero_inv = N/num_in_cat_dict[class3]

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



def calc_weighted_loss(class_weights, y_batch, logits, binary):
    weights = tf.compat.v2.reduce_sum(class_weights * y_batch, axis=1)
    # Categorical cross entropy loss
    
    if binary==True:
        unweighted_losses = tf.compat.v2.nn.sigmoid_cross_entropy_with_logits(y_batch, logits)
    else:
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
    number_of_batches = round(len(indice_set) / batchsize) # get the number of patients in each batch
    index_partitions = [sorted(indice_set[i::number_of_batches]) for i in range(number_of_batches)] # put the indices into batches
    batched_graphs = [[input_matrices[i] for i in index_partition] for index_partition in index_partitions]
    batched_labels = [[labels[i] for i in index_partition] for index_partition in index_partitions]
    return batched_graphs, batched_labels, index_partitions



def load_in_y_event_count():
    """Take in the reduced labels data and create a one-hot encoded label set"""
    event_labels = pd.read_csv('input_data/clinicalevent_service_util_counts.csv')
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

def get_cv_groups(batched_graphs_split1, batched_graphs_split2, batched_graphs_split3, batched_graphs_split4, batched_graphs_split5,
                 batched_labels_split1, batched_labels_split2, batched_labels_split3, batched_labels_split4, batched_labels_split5,
                 cv_split1_indices, cv_split2_indices, cv_split3_indices, cv_split4_indices, cv_split5_indices):
    """
    Takes the splits of the batched graphs and groups them for 5 fold cross validation. Such that each split has 4 in the training group and 1 in the test group.
    Also gets the indices for the batch groups (needed for getting the demographics).
    
    Args:
        list: batched_graphs_splitx - list of SparseTensors for each 5 splits
        list: batched_label_splitx - list of SparseTensors for each 5 splits
        list: cv_splitx_indices - list of batch indices for each 5 splits
    Returns:
        list: graphs_trainx - list of SparseTensors for each of the 5 cv folds
        list: labels_trainx - list of SparseTensors for each of the 5 cv folds
        list: indices_trainx - list of batch indices for each of the 5 cv folds
    """

    graphs_train1 = batched_graphs_split2 + batched_graphs_split3 + batched_graphs_split4 + batched_graphs_split5
    graphs_test1 = batched_graphs_split1
    labels_train1 = batched_labels_split2 + batched_labels_split3 + batched_labels_split4 + batched_labels_split5
    labels_test1 = batched_labels_split1
    indices_train1 = cv_split2_indices + cv_split3_indices + cv_split4_indices + cv_split5_indices
    indices_test1 = cv_split1_indices

    graphs_train2 = batched_graphs_split1 + batched_graphs_split3 + batched_graphs_split4 + batched_graphs_split5
    graphs_test2 = batched_graphs_split2
    labels_train2 = batched_labels_split1 + batched_labels_split3 + batched_labels_split4 + batched_labels_split5
    labels_test2 = batched_labels_split2
    indices_train2 = cv_split1_indices + cv_split3_indices + cv_split4_indices + cv_split5_indices
    indices_test2 = cv_split2_indices

    graphs_train3 = batched_graphs_split1 + batched_graphs_split2 + batched_graphs_split4 + batched_graphs_split5
    graphs_test3 = batched_graphs_split3
    labels_train3 = batched_labels_split1 + batched_labels_split2 + batched_labels_split4 + batched_labels_split5
    labels_test3 = batched_labels_split3
    indices_train3 = cv_split1_indices + cv_split2_indices + cv_split4_indices + cv_split5_indices
    indices_test3 = cv_split3_indices
    

    graphs_train4 = batched_graphs_split1 + batched_graphs_split2 + batched_graphs_split3 + batched_graphs_split5
    graphs_test4 = batched_graphs_split4
    labels_train4 = batched_labels_split1 + batched_labels_split2 + batched_labels_split3 + batched_labels_split5
    labels_test4 = batched_labels_split4
    indices_train4 = cv_split1_indices + cv_split2_indices + cv_split3_indices + cv_split5_indices
    indices_test4 = cv_split4_indices
    

    graphs_train5 = batched_graphs_split1 + batched_graphs_split2 + batched_graphs_split3 + batched_graphs_split4
    graphs_test5 = batched_graphs_split5
    labels_train5 = batched_labels_split1 + batched_labels_split2 + batched_labels_split3 + batched_labels_split4
    labels_test5 = batched_labels_split5
    indices_train5 = cv_split1_indices + cv_split2_indices + cv_split3_indices + cv_split4_indices
    indices_test5 = cv_split5_indices
    


    return graphs_train1, graphs_train2, graphs_train3, graphs_train4, graphs_train5, graphs_test1, graphs_test2, graphs_test3, graphs_test4, graphs_test5, labels_train1, labels_train2, labels_train3, labels_train4, labels_train5, labels_test1, labels_test2, labels_test3, labels_test4, labels_test5, indices_train1, indices_train2, indices_train3, indices_train4, indices_train5, indices_test1, indices_test2, indices_test3, indices_test4, indices_test5


def metric_save(trn_epoch_metric, trn_all_epoch_avgs, 
            val_epoch_metric, val_all_epoch_avgs):
    """
    Save the average epoch to the list of all epochs
    """
    trn_all_epoch_avgs.append(trn_epoch_metric)
    val_all_epoch_avgs.append(val_epoch_metric)
    
    
def get_labels_binary(df):
    """
    Gets the labels in the required format for the model.
    Args:
        df (DataFrame): dataframe containing the labels
    Returns:
        y (DataFrame): binary integer for hip or no replacement.
    """
    def map_values(value):
        if 'hip' in value:
            return 1
        else:
            return 0
    
    df['ohe_binary'] = df['replace_type'].apply(map_values)

    list_of_y = df['ohe_binary'].to_list()
    y = np.array(list_of_y, dtype='float')
    y = tf.constant(y, dtype=tf.float32)
    y = pd.DataFrame({'int_label': list_of_y})   
    
    return y


def check_group_sizes(cv_input_df, test_input_df, cv_y, test_y):
    """
    Check that the size of the inputs match the size of the labels.
    Args:
        cv_input_df (DataFrame): input data for cross validation patients
        cv_y (DataFrame): label data for cross validation patients
        test_input_df (DataFrame): input data for test patients
        test_y (DataFrame): label data for test patients
    """
    
    print(f"Number of people in the cv input data {len(cv_input_df)}")
    print(f"Number of people in cv label data: {cv_y.shape[0]}")

    if len(cv_input_df) != cv_y.shape[0]:
        print("The input and label data size does not match in the cv data.")
        
    print(f"Number of people in the test input data {len(test_input_df)}")
    print(f"Number of people in test label data: {test_y.shape[0]}")

    if len(test_input_df) != test_y.shape[0]:
        print("The input and label data size does not match in the test data.")
        
    print(f"Number of patients in each class in cv set:\n{cv_input_df['replace_type'].value_counts()}\n")
    print(f"Number of patients in each class in test set:\n{test_input_df['replace_type'].value_counts()}")


   
def create_sparse_tensors(sample_size, input_values_indices_df, max_event_codes, max_timesteps, test_or_cv):
    """
    Take the indices and values for each persons temporal graph and then convert them to sparse tensors.
    Args:
        sample_size (int): The number of people in the dataset.
        input_values_indices_df (DataFrame): DataFrame with the columns: 'patientnum','indices','values','num_time_steps'.
        max_event_codes (int): Number of Read codes to use in the model (X and Y of tensor).
        max_timesteps (int): The number of time_steps is the length of the values list. This is needed to create a 
                            sparse matrix.
          
    Returns:
        input_matrices (list of SparseTensors): Representing the temporal graphs
    """
    print(test_or_cv, ": converting from input matrices to SparseTensors")
    input_matrices = []

    for patient in range(sample_size):
        i_list = input_values_indices_df.iloc[patient]['indices'] # indices from patient cell
        v_list = input_values_indices_df.iloc[patient]['values'] # values from patient cell

        individual_sparse = tf.sparse.SparseTensor(i_list, v_list, (max_event_codes, max_event_codes, max_timesteps))

        # Adding the sparse tensor to a list of all the tensors
        ordered_indiv = tf.sparse.reorder(individual_sparse) # reorder required for tensor to work (no effect to outcome)
        input_matrices.append(tf.sparse.expand_dims(ordered_indiv, axis=0))

        if patient%1000 == 0:
            print(f"{patient}/{sample_size} converted to SparseTensors {patient/sample_size:.2%}")
            
    return input_matrices


def convert_demos_to_tensor(df, indices, demo):
    """
    Convert demographics columns for the given batch into a Tensor that can be read into the model.
    Args:
        df (DataFrame): Columns of the demographic data.
        indices (list): Row indices for the current batch.
        demo (boolean): True if demographics are included in this model.
        
    Returns:
        demo_tensor (Tensor): values for demographic data.
    """

    demos = df[['gender', 'imd_quin', 'age_at_label_event']].iloc[indices]
    demos_z = demos.copy()
    demos_z['age_zscore'] = demos_z[['age_at_label_event']].apply(stats.zscore)
    demos_z = demos_z.apply(pd.to_numeric)  
    demo_vals = demos_z[['gender', 'imd_quin', 'age_zscore']].values 
    
    
    demo = demos.apply(pd.to_numeric)
    demo_list = demo[['gender', 'imd_quin', 'age_at_label_event']].values.tolist()
    demo_tensor = tf.convert_to_tensor(demo_vals)

    
    return demo_tensor, demo_list

def calibration_slope(true_y, logits):
    """
    Calculate the calibration slope
    
    Args:
        true_y (list): list of single integer arrays with 0 or 1 depending on the class
        logits (logits tensor): logits for each class
    Returns:
        float: calibration slope value
    """
    # change logits to probabilities
    odds = np.exp(logits)
    np.seterr(invalid='ignore')
    probs = odds / (1 + odds)
    

    fraction_of_pos, mean_pred_value = calibration_curve(true_y, probs, n_bins=10)#, normalize=False)
    try:
        slope, b = np.polyfit(mean_pred_value, fraction_of_pos, 1)

    except:
        slope=0
    
    return slope