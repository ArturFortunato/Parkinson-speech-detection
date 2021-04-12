import os

import pandas     as pd
import statistics as stat

from data_processing import DataProcessing

OPENSMILE   = '/afs/inesc-id.pt/home/aof/opensmile-3.0-linux-x64/bin/SMILExtract'

MFCC_CONF   = '/afs/inesc-id.pt/home/aof/opensmile-3.0-linux-x64/config/mfcc/MFCC12_0_D_A.conf'
GEMAPS_CONF = '/afs/inesc-id.pt/home/aof/opensmile-3.0-linux-x64/config/gemaps/v01b/GeMAPSv01b.conf'
PLP_CONF    = '/afs/inesc-id.pt/home/aof/opensmile-3.0-linux-x64/config/plp/PLP_0_D_A.conf'

class feature_extractor:
    def __list_files(self, path):
        return os.listdir(path)
        
    def __feature_conf_file(self, feature_set):
        if feature_set == 'mfcc':
            return MFCC_CONF
        elif feature_set == 'gemaps':
            return GEMAPS_CONF
        elif feature_set == 'plp':
            return PLP_CONF

    def __clean_csv(self, files, is_control):
        output = []
        dataProcessor = DataProcessing()

        for f in files:
            csv_file = self.read_csv(f)

            #Add name column (filename without extention)
            file_without_ext = f.split("/")[-1].split(".")[0]
            csv_file['name'] = "_".join(file_without_ext.split("_")[:-1])
            
            # Label: 1 for PD, 0 for HC
            csv_file['label'] = 0 if is_control else 1

            output.append(csv_file)

        return output
    
    def __zscore(self, files):
        data_processor = DataProcessing()

        for csv_file in files:
            data_processor.zscore(csv_file, csv_file, columns_to_ignore=['name', 'frameTime', 'label'])
            
    def __extract(self, audios_path, files, type_conf, output_path, conf_file):
        csv_files = []

        for f in files:
            file_without_ext = f.split(".")[0]
            output_file = "{}/{}_{}.csv".format(output_path, file_without_ext, type_conf)
            if not os.path.isfile(output_file):
                os.system("{} -C {} -I {}/{} -csvoutput {}".format(OPENSMILE, conf_file, audios_path, f, output_file))
            
            csv_files.append(output_file)

        return csv_files

    def __average_column(self, data, column_name):
        data = [float(val) for val in data]

        return { "{}_mean".format(column_name): stat.mean(data) }

    def __stdev_column(self, data, column_name):
        data = [float(val) for val in data]

        return { "{}_stdev".format(column_name): stat.stdev(data) }
    
    def avg_stdev_csv(self, filename, columns_to_avg, columns_to_stdev):
        csv_file = pd.read_csv(filename)
        output = {}
        
        if columns_to_avg == [] and columns_to_stdev:
            columns_to_avg = csv_file.columns

        for column in columns_to_avg:
            avg = self.__average_column(csv_file[column], column)
            output = {**output, **avg}

        for column in columns_to_stdev:
            std = self.__stdev_column(csv_file[column], column)
            output = {**output, **std}

        return output
    
    def __concat(self, csvs, output_path):
        combined_csv = pd.concat(csvs, axis=0)

        self.to_csv(combined_csv, output_path)

    def concat_csv(self, csv_files, output_path, index_col=None):
        csvs = [self.read_csv(csv, index_col=index_col) for csv in csv_files]

        self.__concat(csvs, output_path)

    # Merges two csvs by columns
    def merge(self, csv1_file, csv2_file, columns, output_file, how='inner'):
        csv1 = self.read_csv(csv1_file)
        csv2 = self.read_csv(csv2_file)

        output = pd.merge(csv1, csv2, on=columns, how=how)
    
        self.to_csv(output, output_file)

    def to_csv(self, csv, filename, index=False):
        csv.to_csv(filename, index=index, sep=";")

    def read_csv(self, filename, index_col=None):
        return pd.read_csv(filename, sep=";", index_col=index_col)

    def average_and_write_csv(self, input_csv, output_path):
        csv = self.read_csv(input_csv)
        
        # Parses all values to floats. 
        # Ignore options allows to keep 'name' column as a string 
        # without raising an error
        csv = csv.apply(pd.to_numeric, errors='ignore')
        output_csv = csv.groupby(['name']).mean()
        self.to_csv(output_csv, output_path, index=True)

    # is_control is true if group is healthy control
    def extract_features(self, type_conf, audios_path, output_path, is_control):
        wav_files = self.__list_files(audios_path)
        csv_files = self.__extract(audios_path, wav_files, type_conf, output_path, self.__feature_conf_file(type_conf))
        self.__zscore(csv_files)
        csv_list  = self.__clean_csv(csv_files, is_control)
        
        
        output_file = "{}/{}_{}.csv".format(output_path, type_conf, "HC" if is_control else "PD")
        self.__concat(csv_list, output_file)

        return output_file
