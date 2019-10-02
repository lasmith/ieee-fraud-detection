import random
import numpy as np
from catboost import Pool
from pandas import DataFrame

from data.preprocessor import get_categorical_cols


def get_x_y(df):
    """
    Split the dataframe into the features and the target
    :param df: The data frame
    :return: X, y - The features and the target respectively
    """
    X = df.drop('isFraud', axis=1)
    y = df.isFraud
    return X, y

def gen_seeds(seed=0):
    '''
    Ensure seeds are set on anything that needs it.
    Passing in the same value here would make the process deterministic
    '''
    random.seed(seed)
    np.random.seed(seed)

def get_lbo_pools(df_train:DataFrame):
    """
    Split the data frame into a last block out by date. The training data is split by time so that the last month
    is used as the validation set. This is a common practice for time series as going forward you will have to predict
    new data as it comes in.
    :param df_train: The data frame to split
    :param categorical_features_indices: The column indexes of the categorical
    :return: train_pool, validate_pool - Catboost Pool's representing the training / validation data set
    """
    main_train_set = df_train[df_train['DT_M']<(df_train['DT_M'].max())].reset_index(drop=True)
    validation_set = df_train[df_train['DT_M']==df_train['DT_M'].max()].reset_index(drop=True)
    print ("Training shape: %s, validation shape: %s"%(main_train_set.shape, validation_set.shape))
    X, y = get_x_y(main_train_set)
    X_valid, y_valid = get_x_y(validation_set)
    return X, y, X_valid, y_valid

def get_catboost_pools(X, y, X_valid, y_valid):
    """
    Convert data sets into catboost specific Pool's
    :param X: The training feature set
    :param y: The training target (to predict)
    :param X_valid: The validation feature set
    :param y_valid: The validation target
    :return: train_pool, validate_pool - Catboost wrappers around the training / validation data
    """
    categorical_features_indices = get_categorical_cols(X)
    train_pool = Pool(X, y, cat_features=categorical_features_indices)
    validate_pool = Pool(X_valid, y_valid, cat_features=categorical_features_indices)
    return train_pool, validate_pool
