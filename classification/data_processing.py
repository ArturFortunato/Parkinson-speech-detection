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
    def zscore(self, input_csv, output_path, columns_to_ignore=[]):
        csv = pd.read_csv(input_csv, sep=";")
        for column in csv.columns:
            if column in columns_to_ignore:
                continue
            csv[column] = stats.zscore(csv[column])
        csv.to_csv(output_path, sep=";", index=False)
