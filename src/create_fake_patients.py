import pandas as pd
import random

def create_fake_index_list(max_events, max_nodes):
    """Create indices for fake patients e.g. [[56,23,99], [34,3,98]]
    such that each sublist would be [start_node, end_node, timestep].

    Returns:
        list of lists: indices for 3D matrices.
    """

    # minimum number of events needs to be 2 to make a graph
    min_num_events = 2

    indices_list_of_lists = []
    for i in range(random.randint(min_num_events, max_events)):
        if i!= 0:
            indices_list_of_lists.append([random.randint(0, max_nodes-1),indices_list_of_lists[i-1][0],max_events-i])
        else:
            indices_list_of_lists.append([random.randint(0, max_nodes-1),random.randint(0, max_nodes-1),max_events-i])
    
    return indices_list_of_lists


def create_fake_patient_df(num_patients, max_events, max_nodes):
    """Create df with columns: User number | Indices (int, list of lists) |
    Values (list of floats) | Num time steps (int) | gender (bin int)

    Args:
        num_patients (int): number of patient rows to generate
        max_events (int): maximum number of events/visits a 'patient' can have
        max_nodes (int): maximum number of different types of nodes/read codes that can be used

    Returns:
        dataframe: df with columns for inputs and labels
    """
    # create a dictionary with the index as keys and the values as lists
    data = {'user': [i for i in range(1, num_patients)],
            'indices': [create_fake_index_list(max_events, max_nodes) for i in range(1, num_patients)],
            'values':0,
            'num_time_steps':0,
            'gender':0,
            'imd_quin':0,
            'age_at_label_event':0,
            'replace_type': 'n',
            'indices_len':0
            }

    # create a Pandas DataFrame from the dictionary
    df = pd.DataFrame(data)

    for row_num, row in df.iterrows():
        indices_list = df.iloc[row_num,1] 
        num_timesteps = len([elem for elem in indices_list])
        df.iloc[row_num, 3] = num_timesteps

    values_list = []
    gender_list = []
    imd_list = []
    age_list = []
    replace_list = []
    for row_num, row in df.iterrows():
        num_timesteps = df.iloc[row_num,3]
        values_list.append([random.uniform(0.0, 1.0) for i in range(num_timesteps)])
        gender_list.append(random.randint(0, 1))
        imd_list.append(float(random.randint(0, 4))+1.0)
        age_list.append(float(random.randint(39, 89))+1.0)
        replace_list.append(random.choice(['hip', 'none']))

    df['values'] = values_list
    df['gender'] = gender_list
    df['imd_quin'] = imd_list
    df['age_at_label_event'] = age_list
    df['replace_type'] = replace_list
    df['indices_len'] = df['indices'].apply(lambda x: len(x))

    return df




def create_fake_int_label(num_patients):
    """ Create a dataframe with one column randomly selecting 0 to 2 of size num_patients,
    to represent patient label.

    Args:
        num_patients (int): Number of patients (needs to be the same as the input number)

    Returns:
        dataframe: One column dataframe with an integer between 0 and 3.
    """
    
    # create a list of random integers between 0 and 2 (inclusive)
    random_values = [random.randint(0, 2) for i in range(num_patients)]

    # create a Pandas DataFrame with one column containing the random values
    df = pd.DataFrame({'int_label': random_values})

    return df
