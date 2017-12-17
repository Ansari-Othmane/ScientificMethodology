from datetime import timedelta
from datetime import datetime
from pandas_datareader import data
%matplotlib inline  
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def getData(stock_name, start_date, end_date, event_date, source='yahoo'):
    # Convert date parameters into datetime type
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    event = datetime.strptime(event_date, '%Y-%m-%d')
    
    # Get data using an API
    ts = data.DataReader(stock_name, source, start, end)['Adj Close']
    
    '''
    Get the number of days we have in our for our training+validation, 
    then have the training be on 80% of these, the validation on 20%
    '''
    train_days = (event - start).days
    inter_date = start+timedelta(days=int(train_days*0.8))
    ts_train = ts[start:inter_date]
    ts_validate = ts[inter_date+timedelta(days=1):event-timedelta(days=1)]
    ts_test = ts[event:]
    
    return ts_train, ts_validate, ts_test

def plotData(data,x_label,y_label,title):
    plt.rcParams["figure.figsize"] = [15.0,10.0]
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend('price legend')
    data.plot()
    
    
def gainPercentage(ts):
    ts_gain = pd.DataFrame(columns=['gain'])
    
    # Get the data starting from the 2nd element
    data = ts[1:]
    
    '''
    The formula of the gain being the current stock price minus the previous one, devided by the latter, 
    we'll need to keep track of the previous index at all times
    '''
    previous_index = ts.index[0]
    
    # Iterating over the data without the  first element, and compute the gain using the aforementioned formula
    for index_ts, row_ts in data.iteritems():
        gain_value = (ts[index_ts]-ts[previous_index])/ts[previous_index]
        ts_gain.loc[index_ts] = gain_value
        previous_index = index_ts
    
    return ts_gain.gain

def dropOutliers(train_data,sd_coef):
    data_gain = gainPercentage(train_data)
    mean_gain = np.mean(data_gain.values)
    risk = np.std(data_gain.values)
    
    for gain,index in zip(data_gain.values,data_gain.index):
        if gain > mean_gain + (risk*sd_coef) or gain < mean_gain - (risk*sd_coef):
            if data_gain[0] == gain:
                data_gain[0] = data_gain[1]
            else :
                if data_gain[-1] == gain:
                    data_gain[-1] = data_gain[-2]
                else:
                    prev_val = data_gain.shift(1)[data_gain.index.get_loc(index)]
                    next_val = data_gain.shift(-1)[data_gain.index.get_loc(index)]
                    data_gain[data_gain.index.get_loc(index)] = (prev_val + next_val ) / 2
        
    return data_gain
    
    
    

