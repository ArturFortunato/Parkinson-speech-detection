import os

import pandas     as pd
import statistics as stat

OPENSMILE      = '/afs/inesc-id.pt/home/aof/opensmile-3.0-linux-x64/bin/SMILExtract'
MFCC_CONF      = '/afs/inesc-id.pt/home/aof/opensmile-3.0-linux-x64/config/mfcc/MFCC12_0_D_A.conf'
MFCC_CONF_MEAN = '/afs/inesc-id.pt/home/aof/opensmile-3.0-linux-x64/config/mfcc/MFCC12_E_D_A_Z.conf'

class feature_extractor:
    def __list_files(self, path):
        return os.listdir(path)
        
    def __clean_csv(self, files, is_control):
        output = []

        for f in files:
            csv_file = pd.read_csv(f, sep=";")

            #Remove frameTime column
            csv_file = csv_file.drop(['frameTime'], axis=1)
                    
            #Add name column (filename without extention
            file_without_ext = f.split("/")[-1].split(".")[0]
            csv_file['name'] = file_without_ext
            
            # Label: 1 for PD, 0 for HC
            csv_file['label'] = 0 if is_control else 1

            output.append(csv_file)

        return output

    def __extract(self, audios_path, files, type_conf, output_path, conf_file):
        csv_files = []

        for f in files:
            file_without_ext = f.split(".")[0]
            output_file = "{}/{}_{}.csv".format(output_path, file_without_ext, type_conf)
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

        for column in columns_to_avg:
            avg = self.__average_column(csv_file[column], column)
            output = {**output, **avg} 

        for column in columns_to_stdev:
            std = self.__stdev_column(csv_file[column], column)
            output = {**output, **std} 

        return output
    
    def __merge(self, csvs, output_path):
        combined_csv = pd.concat(csvs)

        combined_csv.to_csv(output_path, index=False, sep=";")

    def merge_csv(self, csv_files, output_path):
        csvs = [pd.read_csv(csv, sep=";") for csv in csv_files]

        self.__merge(csvs, output_path)

    # is_control is true if group is healthy control
    def extract_features(self, type_conf, audios_path, output_path, is_control):
        wav_files = self.__list_files(audios_path)
        csv_files = self.__extract(audios_path, wav_files, type_conf, output_path, MFCC_CONF)
        csv_list  = self.__clean_csv(csv_files, is_control)
        
        output_file = "{}/mfcc_{}.csv".format(output_path, "HC" if is_control else "PD")
        self.__merge(csv_list, output_file)

        return output_file
