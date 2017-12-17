from pandas_datareader import data
%matplotlib inline  
import matplotlib.pyplot as plt
import pandas as pd



def plotData(data,x_label,y_label,title):
    plt.rcParams["figure.figsize"] = [15.0,10.0]
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend('price legend')
    data.plot()
    
    
    

