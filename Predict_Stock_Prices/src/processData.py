import pandas as pd
import numpy as np
import math
import sys

# Import sklearn.preprocessing.StandardScaler
from sklearn.preprocessing import MinMaxScaler


def normalise_data(data):
    """
    Normalises the data values using MinMaxScaler from sklearn
    :param data: a DataFrame with columns as  ['index','Open','Close','Volume']
    :return: a DataFrame with normalised value for all the columns except index
    """
    # Initialize a scaler, then apply it to the features
    scaler = MinMaxScaler()
    numerical = ['Open', 'Close', 'Volume']
    data[numerical] = scaler.fit_transform(data[numerical])

    return data


def remove_data(data):
    """
    Remove columns from the data
    :param data: a record of all the stock prices with columns as  ['Date','Open','High','Low','Close','Volume']
    :return: a DataFrame with columns as  ['index','Open','Close','Volume']
    """
    # Define columns of data to keep from historical stock data
    item = []
    open = []
    close = []
    volume = []

    # Loop through the stock data objects backwards and store factors we want to keep
    i_counter = 0
    for i in range(0, len(data), 1):
        item.append(i_counter)
        open.append(data['Open'][i])
        close.append(data['Close'][i])
        volume.append(data['Volume'][i])
        i_counter += 1

    # Create a data frame for stock data
    stocks = pd.DataFrame()

    # Add factors to data frame
    stocks['Item'] = item
    stocks['Open'] = open
    stocks['Close'] = close
    stocks['Volume'] = volume

    # return new formatted data
    return stocks


def series_to_supervised(dataset, history_size, target_size):
    """
        Split the data set into tr aining and testing feature for Long Short Term Memory Model
        :param data: whole data set containing ['Item','Open','Close','Volume'] features
        :param history_size: no of days
        :param target_size: size of test data to be used
        :return: X: sets of feature
        :return: y: sets of label
    """
    dataset.columns = ['Item','Open','Close','Volume']
#     dataset = dataset.drop(['Item'], axis=1)
    dataset = dataset['Close']
    dataset = dataset.values
    
    start_index = history_size
    end_index = len(dataset) - target_size
    
    features = []
    labels   = []
    
    for i in range(start_index, end_index):
        indices = range(i - history_size, i, 1)
        features.append(dataset[indices])
        labels.append(dataset[i:i + target_size])
        
    features = np.array(features)
    labels   = np.array(labels)
    return features, labels


def train_test_split(stocks, train_frac, *args, **kwargs):
    """
        Split the data set into training and testing feature for Linear Regression Model
        :param stocks: whole data set containing ['Open','Close','Volume'] features
        :return: X_train : training sets of feature
        :return: X_test : test sets of feature
        :return: y_train: training sets of label
        :return: y_test: test sets of label
    """
    # Extract the validation size if provided
    val_frac = kwargs.get('val_frac', None)
    
    assert (train_frac < 1.0),"Training size exceeds dataset dimensions. Choose training fraction < 1.0"
        
    # convert the df into a matrix for ease of splitting
    stock_matrix = stocks.as_matrix()

    # Define Test/Train Split 80/20
    train_size = int(stock_matrix.shape[0] * train_frac)
    
    if not val_frac is None:
        val_size = int(train_size * val_frac)
        
    # Set up training, validation and test sets
    selector = [x for x in range(stock_matrix.shape[1]) if x != 2]
    
    if not val_frac is None:
        # train dataset
        X_train = stock_matrix[:(train_size-val_size), selector]
        y_train = stock_matrix[:(train_size-val_size), 2]
        # validation dataset
        X_val = stock_matrix[(train_size-val_size):train_size, selector]
        y_val = stock_matrix[(train_size-val_size):train_size, 2]
        # test dataset
        X_test = stock_matrix[train_size:, selector]
        y_test = stock_matrix[train_size:, 2]
        # return the datasets
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    else:
        # train dataset
        X_train = stock_matrix[:train_size, selector]
        y_train = stock_matrix[:train_size, 2]
        # test dataset
        X_test = stock_matrix[train_size:, selector]
        y_test = stock_matrix[train_size:, 2]
        # return the datasets
        return X_train, X_test, y_train, y_test
    

def create_dataset_lstm(stocks, train_frac=0.8, history_size=1, target_size=1, **kwargs):
    """
        Split the data set into tr aining and testing feature for Long Short Term Memory Model
        :param stocks: whole data set containing ['Open','Close','Volume'] features
        :param train_frac: fraction of dataset used for traiing. Used to invoke train_test_spli()
        :param history_size: no of days
        :param target_size: size of test data to be used
        :return: X_train : training sets of feature
        :return: X_val: validation sets of feature
        :return: X_test : test sets of feature
        :return: y_train: training sets of label
        :return: y_val : validation sets of label
        :return: y_test: test sets of label
    """
    # Extract the validation size if provided
    val_frac = kwargs.get('val_frac', None)
    
    # training data
    if val_frac is not None:
        
        X_train, X_val, X_test, y_train, y_val, y_test = train_test_split(stocks, 
                                                                          train_frac,
                                                                          val_frac=val_frac)
        
        val = pd.concat([pd.DataFrame(X_val), pd.DataFrame(y_val)], axis=1)
        train = pd.concat([pd.DataFrame(X_train), pd.DataFrame(y_train)], axis=1)
        test = pd.concat([pd.DataFrame(X_test), pd.DataFrame(y_test)], axis=1)
        
        val_X, val_y     = series_to_supervised(val, history_size, target_size)
        train_X, train_y = series_to_supervised(train, history_size, target_size)
        test_X, test_y   = series_to_supervised(test, history_size, target_size)
        
        return train_X, val_X, test_X, train_y, val_y, test_y
        
    else:
        
        X_train, X_test, y_train, y_test = train_test_split(stocks, train_frac)
        
        train = pd.concat([pd.DataFrame(X_train), pd.DataFrame(y_train)], axis=1)
        test = pd.concat([pd.DataFrame(X_test), pd.DataFrame(y_test)], axis=1)
        
        train_X, train_y = series_to_supervised(train, history_size, target_size)
        test_X, test_y   = series_to_supervised(test, history_size, target_size)
        
        return train_X, test_X, train_y, test_y
        