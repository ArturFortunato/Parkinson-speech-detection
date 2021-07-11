import os 

import opensmile
import parselmouth
from parselmouth.praat import call

import pandas as pd

# Male   pitch ranges from 60  to 180 Hz
# Female pitch ranges from 160 t0 300 Hz
# These values are defined to include the  
# whole spectrum of human voices
LOWER_PITCH = 60
UPPER_PITCH = 300
UNIT = "Hertz"

def verbose_opensmile():
    for fset in opensmile.FeatureSet:
        print("========================================================")
        print(fset)    
        for level in opensmile.FeatureLevel:
            print("==========================")
            print(level)
            try:
                smile = opensmile.Smile(
                    feature_set=fset,
                    feature_level=level,
                )
                print(smile.feature_names)
            except:
                pass

class praat_extractor:
    
    # Extracts HNR, mean and avg F0, localJitter, localShimmer
    def __basic_features(self, row, path, subfolders):
        print(row["name"])
        sound = parselmouth.Sound("{}/{}/{}.wav".format(path, subfolders[row['label']], row['name']))
        #sound = parselmouth.Sound("{}/{}/{}.wav".format(path, subfolders[row['label']], "_".join(row['name'].split("_")[:-1])))
        
        harmonicity = sound.to_harmonicity()

        #This produces a square matrix for GNR, not quite sure what it is
        #harmonicity_gne = sound.to_harmonicity_gne()

        pitch = call(sound, "To Pitch", 0.0, LOWER_PITCH, UPPER_PITCH)
        pointProcess = call(sound, "To PointProcess (periodic, cc)", LOWER_PITCH, UPPER_PITCH)

        meanF0 = call(pitch, "Get mean", 0, 0, UNIT)
        stdevF0 = call(pitch, "Get standard deviation", 0 ,0, UNIT)
        
        localJitter  = call(pointProcess, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        localShimmer = call([sound, pointProcess], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)

        hnr = harmonicity.values[harmonicity.values != -200].mean()

        return [hnr, meanF0, stdevF0, localJitter, localShimmer]

    def add_hnr(self, subfolders, csv_in, csv_out, path):
    
        dataframe = pd.read_csv(csv_in, sep=";")
        # Apply parselmouth wrapper function row-wise
        #hnr, meanF0, stdevF0, jitter, shimmer = dataframe.apply(self.__basic_features, path=path, subfolders=subfolders, axis='columns')
        dataframe[['hnr', 'meanF0', 'stdevF0', 'jitter', 'shimmer']] = dataframe.apply(self.__basic_features, path=path, subfolders=subfolders, axis='columns', result_type='expand')
        print(dataframe.columns)
        input()
        dataframe.to_csv(csv_out, index=False, sep=";")
