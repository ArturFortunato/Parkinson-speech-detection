import pandas as pd
from scipy import stats

'''
    This class contains multiple methods to 
    process input data in order to achieve
    a higher classificaion performance
'''
class DataProcessing():

    '''
        Data will be normalized by column (feature)
        Final mean will be 0 and stdev will be 1
    '''
    def zscore(self, csv, columns_to_ignore=[]):
        for column in csv.columns:
            if column in columns_to_ignore:
                print("Ignoring column {}".format(column))
                continue
            csv[column] = stats.zscore(csv[column])
        return csv
