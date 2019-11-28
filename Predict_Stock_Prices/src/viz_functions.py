
# Some header : 
#%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['figure.figsize'] = (18, 12)
plt.rc('xtick', labelsize=12)     
plt.rc('ytick', labelsize=12)


def price(x):
    """
    format the coords message box
    :param x: data to be formatted
    :return: formatted data
    """
    return '$%1.2f' % x


def plot_basic(stocks, title='Google Trading', y_label='Price USD', x_label='Trading Days', **kwargs):
    """
    Plots basic pyplot
    :param stocks: DataFrame having all the necessary data
    :param title:  Title of the plot 
    :param y_label: yLabel of the plot
    :param x_label: xLabel of the plot
    :return: prints a Pyplot againts items and their closing value
    """
    filename = kwargs.get('filename', None)
    
    fig, ax = plt.subplots()
    ax.plot(stocks['Item'], stocks['Close'], '#34495E', linewidth=2.5, linestyle="-")
    #0A7388
    ax.format_ydata = price
    ax.set_title(title, size = 18)

    # Add labels
    plt.ylabel(y_label, size = 16)
    plt.xlabel(x_label, size = 16)

#     plt.show()
    fig.savefig(filename, dpi=300, bbox_inches='tight')


def plot_prediction(actual, prediction, title='Google Trading vs Prediction', y_label='Price USD', x_label='Trading Days', **kwargs):
    """
    Plots train, test and prediction
    :param actual: DataFrame containing actual data
    :param prediction: DataFrame containing predicted values
    :param title:  Title of the plot
    :param y_label: yLabel of the plot
    :param x_label: xLabel of the plot
    :return: prints a Pyplot againts items and their closing value
    """
    filename = kwargs.get('filename', None)
#     leg_prop = font_manager.FontProperties(size=14)
#     title_prop = font_manager.FontProperties(size=18)
    
    fig, ax = plt.subplots()
#     fig = plt.figure()
#     ax = fig.add_subplot(111)

    # Add labels
    plt.ylabel(y_label, size = 16)
    plt.xlabel(x_label, size = 16)

    # Plot actual and predicted close values

    ax.plot(actual, '#D35400', label='Adjusted Close', linewidth=2.5, linestyle="-")
    ax.plot(prediction, '#3498DB', label='Predicted Close', linewidth=2.5, linestyle="-")

    # Set title
    ax.set_title(title, size = 18)
    ax.legend(loc='upper left')

    fig.savefig(filename, dpi=300, bbox_inches='tight')