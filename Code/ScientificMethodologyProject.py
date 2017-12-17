from pandas_datareader import data
%matplotlib inline  
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np



def plotData(data,x_label,y_label,title):
    plt.rcParams["figure.figsize"] = [15.0,10.0]
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend('price legend')
    data.plot()
    
    
def gainPercentage(ts):
    ts_gain = pd.DataFrame(columns=['gain'])
    data = ts[1:]
    previous_index = ts.index[0]
    for index_ts, row_ts in data.iteritems():
        gain_value = (ts[previous_index]-ts[index_ts])/ts[previous_index]
        ts_gain.loc[index_ts] = gain_value
        previous_index = index_ts
    return ts_gain.gain
    
    
    

