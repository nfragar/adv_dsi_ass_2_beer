# A Sets file that has dataset functions and classes

# Author: Anthony So

def subset_x_y(target, features, start_index:int, end_index:int):
    """Keep only the rows for X and y sets from the specified indexes

    Parameters
    ----------
    target : pd.DataFrame
        Dataframe containing the target
    features : pd.DataFrame
        Dataframe containing all features
    features : int
        Index of the starting observation
    features : int
        Index of the ending observation

    Returns
    -------
    pd.DataFrame
        Subsetted Pandas dataframe containing the target
    pd.DataFrame
        Subsetted Pandas dataframe containing all features
    """
    
    return features[start_index:end_index], target[start_index:end_index]

def split_sets_by_time(df, target_col, test_ratio=0.2):
    """Split sets by indexes for an ordered dataframe

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    target_col : str
        Name of the target column
    test_ratio : float
        Ratio used for the validation and testing sets (default: 0.2)

    Returns
    -------
    Numpy Array
        Features for the training set
    Numpy Array
        Target for the training set
    Numpy Array
        Features for the validation set
    Numpy Array
        Target for the validation set
    Numpy Array
        Features for the testing set
    Numpy Array
        Target for the testing set
    """
    
    df_copy = df.copy()
    target = df_copy.pop(target_col)
    cutoff = int(len(target) / 5)
    
    X_train, y_train = subset_x_y(target=target, features=df_copy, start_index=0, end_index=-cutoff*2)
    X_val, y_val     = subset_x_y(target=target, features=df_copy, start_index=-cutoff*2, end_index=-cutoff)
    X_test, y_test   = subset_x_y(target=target, features=df_copy, start_index=-cutoff, end_index=len(target))

    return X_train, y_train, X_val, y_val, X_test, y_test

def save_sets(X_train=None, y_train=None, X_val=None, y_val=None, X_test=None, y_test=None, path='../data/processed/'):
    """Save the different sets locally

    Parameters
    ----------
    X_train: Numpy Array
        Features for the training set
    y_train: Numpy Array
        Target for the training set
    X_val: Numpy Array
        Features for the validation set
    y_val: Numpy Array
        Target for the validation set
    X_test: Numpy Array
        Features for the testing set
    y_test: Numpy Array
        Target for the testing set
    path : str
        Path to the folder where the sets will be saved (default: '../data/processed/')

    Returns
    -------
    """
    import numpy as np

    if X_train is not None:
      np.save(f'{path}X_train', X_train)
    if X_val is not None:
      np.save(f'{path}X_val',   X_val)
    if X_test is not None:
      np.save(f'{path}X_test',  X_test)
    if y_train is not None:
      np.save(f'{path}y_train', y_train)
    if y_val is not None:
      np.save(f'{path}y_val',   y_val)
    if y_test is not None:
      np.save(f'{path}y_test',  y_test)


def load_sets(path='../data/processed/', val=False):
    """Load the different locally save sets

    Parameters
    ----------
    path : str
        Path to the folder where the sets are saved (default: '../data/processed/')

    Returns
    -------
    Numpy Array
        Features for the training set
    Numpy Array
        Target for the training set
    Numpy Array
        Features for the validation set
    Numpy Array
        Target for the validation set
    Numpy Array
        Features for the testing set
    Numpy Array
        Target for the testing set
    """
    import numpy as np
    import os.path

    X_train = np.load(f'{path}X_train.npy') if os.path.isfile(f'{path}X_train.npy') else None
    X_val   = np.load(f'{path}X_val.npy'  ) if os.path.isfile(f'{path}X_val.npy')   else None
    X_test  = np.load(f'{path}X_test.npy' ) if os.path.isfile(f'{path}X_test.npy')  else None
    y_train = np.load(f'{path}y_train.npy') if os.path.isfile(f'{path}y_train.npy') else None
    y_val   = np.load(f'{path}y_val.npy'  ) if os.path.isfile(f'{path}y_val.npy')   else None
    y_test  = np.load(f'{path}y_test.npy' ) if os.path.isfile(f'{path}y_test.npy')  else None
    
    return X_train, y_train, X_val, y_val, X_test, y_test

def pop_target(df, target_col, to_numpy=False):
    """Extract target variable from dataframe and convert to nympy arrays if required

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe
    target_col : str
        Name of the target variable
    to_numpy : bool
        Flag stating to convert to numpy array or not

    Returns
    -------
    pd.DataFrame/Numpy array
        Subsetted Pandas dataframe containing all features
    pd.DataFrame/Numpy array
        Subsetted Pandas dataframe containing the target
    """

    df_copy = df.copy()
    target = df_copy.pop(target_col)
    
    if to_numpy:
        df_copy = df_copy.to_numpy()
        target = target.to_numpy()
    
    return df_copy, target

# Source: AdvDSI-Lab3-Exercise1-Solutions.ipynb

def split_sets_random(df, target_col, test_ratio=0.2, to_numpy=False, stratify_dataset= None, reduce_dataset=False, reduce_ratio=0.2):
    """Split sets randomly

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    target_col : str
        Name of the target column
    test_ratio : float
        Ratio used for the validation and testing sets (default: 0.2)
    stratify_dataset : boolean
        Determines if dataset is stratefied based upon target
    reduce_dataset : boolean
        Determines if dataset is stratefied based upon target        
    reduce_ratio : float
        Ratio used to reduce the overall dataset (default: 0.2)        

    Returns
    -------
    Numpy Array
        Features for the training set
    Numpy Array
        Target for the training set
    Numpy Array
        Features for the validation set
    Numpy Array
        Target for the validation set
    Numpy Array
        Features for the testing set
    Numpy Array
        Target for the testing set
    Numpy Array
        Features for the remaining dataset
    Numpy Array
        Target for the remaining dataset
    """
    from sklearn.model_selection import train_test_split
    
    # Pop the target column
    features, target = pop_target(df=df, target_col=target_col, to_numpy=to_numpy)
    
    if reduce_dataset == True:
        
        # Use Target Column to stratify (if required)
        if stratify_dataset == None:
            stratify_dataset_input = None
        else:
            features_remaining = features
            target_remaining = target
            stratify_dataset_input = target_remaining
            
        # Split Data 
        X_remaining, features, y_remaining, target = train_test_split(features_remaining, target_remaining, test_size=reduce_ratio, random_state=8, stratify=stratify_dataset_input)
        
    

    # Use Target Column to stratify (if required)
    if stratify_dataset == None:
        stratify_dataset_input = None
    else:
        stratify_dataset_input = target
    
    # Split Data 
    X_data, X_test, y_data, y_test = train_test_split(features, target, test_size=test_ratio, random_state=8, stratify=stratify_dataset_input)
    
    val_ratio = test_ratio / (1 - test_ratio)
    
    # Use y_data Column to stratify (if required)
    if stratify_dataset == None:
        stratify_dataset_input = None
    else:
        stratify_dataset_input = y_data
    
    X_train, X_val, y_train, y_val = train_test_split(X_data, y_data, test_size=val_ratio, random_state=8, stratify=stratify_dataset_input)

    return X_train, y_train, X_val, y_val, X_test, y_test, X_remaining, y_remaining
