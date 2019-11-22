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

def scale_range(x, input_range, target_range):
    """
    Rescale a numpy array from input to target range
    :param x: data to scale
    :param input_range: optional input range for data: default 0.0:1.0
    :param target_range: optional target range for data: default 0.0:1.0
    :return: rescaled array, incoming range [min,max]
    """

    range = [np.amin(x), np.amax(x)]
    x_std = (x - input_range[0]) / (1.0*(input_range[1] - input_range[0]))
    x_scaled = x_std * (1.0*(target_range[1] - target_range[0])) + target_range[0]
    return x_scaled, range

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
    

def train_test_split_lstm(stocks, prediction_time=1, test_data_size=450, unroll_length=50):
    """
        Split the data set into training and testing feature for Long Short Term Memory Model
        :param stocks: whole data set containing ['Open','Close','Volume'] features
        :param prediction_time: no of days
        :param test_data_size: size of test data to be used
        :param unroll_length: how long a window should be used for train test split
        :return: X_train : training sets of feature
        :return: X_test : test sets of feature
        :return: y_train: training sets of label
        :return: y_test: test sets of label
    """
    # training data
    test_data_cut = test_data_size + unroll_length + 1

    x_train = stocks[0:-prediction_time - test_data_cut].as_matrix()
    y_train = stocks[prediction_time:-test_data_cut]['Close'].as_matrix()

    # test data
    x_test = stocks[0 - test_data_cut:-prediction_time].as_matrix()
    y_test = stocks[prediction_time - test_data_cut:]['Close'].as_matrix()

    return x_train, x_test, y_train, y_test


def unroll(data, sequence_length=24):
    """
    use different windows for testing and training to stop from leak of information in the data
    :param data: data set to be used for unrolling
    :param sequence_length: window length
    :return: data sets with different window.
    """
    result = []
    for index in range(len(data) - sequence_length):
        result.append(data[index: index + sequence_length])
    return np.asarray(result)
