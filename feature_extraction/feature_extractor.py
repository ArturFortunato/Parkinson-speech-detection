import os

import pandas     as pd
import statistics as stat

from data_processing import DataProcessing

OPENSMILE   = '/afs/inesc-id.pt/home/aof/opensmile-3.0-linux-x64/bin/SMILExtract'

MFCC_CONF       = '/afs/inesc-id.pt/home/aof/opensmile-3.0-linux-x64/config/mfcc/MFCC12_0_D_A.conf'
GEMAPS_CONF     = '/afs/inesc-id.pt/home/aof/opensmile-3.0-linux-x64/config/gemaps/v01b/GeMAPSv01b.conf'
PLP_CONF        = '/afs/inesc-id.pt/home/aof/opensmile-3.0-linux-x64/config/plp/PLP_0_D_A.conf'
PROSODY_CONF    = '/afs/inesc-id.pt/home/aof/opensmile-3.0-linux-x64/config/prosody/prosodyAcf2.conf'

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
        elif feature_set == 'prosody':
            return PROSODY_CONF

    def __clean_csv(self, files, is_control, columns_to_use=None):
        output = []
        data_processing = DataProcessing()

        for f in files:
            print('file')
            print(f)
            csv_file = self.read_csv(f)

            #Add name column (filename without extention)
            file_without_ext = f.split("/")[-1].split(".")[0]
            csv_file['name'] = "_".join(file_without_ext.split("_")[:-1])
            
            # Label: 1 for PD, 0 for HC
            csv_file['label'] = 0 if is_control else 1

            if columns_to_use is not None:
                csv_file = csv_file.loc[:, csv_file.columns.isin(columns_to_use)]
            
            # Zscore all data columns
            #csv_file = data_processing.zscore(csv_file, columns_to_ignore=['label', 'name', 'frameTime'])
            
            output.append(csv_file)

        return output
    
    def __get_opensmile_instruction(self, conf_file, audios_path, f, output_file, lld):
        if lld:
            print('lld')
            print("{} -C {} -I {}/{} -lldcsvoutput {}".format(OPENSMILE, conf_file, audios_path, f, output_file))
            return "{} -C {} -I {}/{} -lldcsvoutput {}".format(OPENSMILE, conf_file, audios_path, f, output_file)

        return "{} -C {} -I {}/{} -csvoutput {}".format(OPENSMILE, conf_file, audios_path, f, output_file)

    def __extract(self, audios_path, files, type_conf, output_path, conf_file, lld, single_file=None):
        csv_files = []

        if single_file is not None:
            output_file = 'single_subject.csv'
            print(audios_path)
            os.system(self.__get_opensmile_instruction(conf_file, '', files[0], output_file, lld))

            csv_files.append(output_file)

        else: 
            for f in files:
                file_without_ext = f.split(".")[0]
                output_file = "{}/{}_{}.csv".format(output_path, file_without_ext, type_conf)
                if not os.path.isfile(output_file):
                    os.system(self.__get_opensmile_instruction(conf_file, audios_path, f, output_file, lld))
            
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

    # Merges a list of csvs by columns
    def merge(self, csvs, columns, output_file, how='inner'):
        if len(csvs) == 0:
            print("At least 1 csv must exist in order to merge")
            exit(-1)

        output = self.read_csv(csvs[0])

        for csv in csvs[1:]:
            output = pd.merge(output, self.read_csv(csv), on=columns, how=how).drop_duplicates()

        print(output.columns)
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
    def extract_features(self, type_conf, audios_path, output_path, is_control, columns_to_use=None, audio_file=None):
        wav_files = self.__list_files(audios_path) if audio_file is None else [audio_file]
        csv_files = self.__extract(audios_path, wav_files, type_conf, output_path, self.__feature_conf_file(type_conf), type_conf == "gemaps", single_file=audio_file)
        csv_list  = self.__clean_csv(csv_files, is_control, columns_to_use=columns_to_use)
        
        
        output_file = "{}/{}_{}.csv".format(output_path, type_conf, "HC" if is_control else "PD")
        self.__concat(csv_list, output_file)

        return output_file

