import os

from feature_extractor import feature_extractor
from praat_extractor   import praat_extractor

import paths

def extract_opensmile(extractor, dataset, output_path, feature_group, subfolders, audios_path):
    if not os.path.isfile("{}/{}_{}.csv".format(output_path, dataset, feature_group)):
        hc_csv = extractor.extract_features(feature_group, "{}/{}".format(audios_path, subfolders[0]), output_path, True )
        pd_csv = extractor.extract_features(feature_group, "{}/{}".format(audios_path, subfolders[1]), output_path, False)

        extractor.merge_csv([hc_csv, pd_csv], "{}/{}_{}.csv".format(output_path, dataset, feature_group))
        
    if not os.path.isfile("{}/{}_{}_avg.csv".format(output_path, dataset, feature_group)):
        extractor.average_and_write_csv("{}/{}_{}.csv".format(output_path, dataset, feature_group), "{}/{}_{}_avg.csv".format(output_path, dataset, feature_group))

def extract_praat(extractor, subfolders, csv_in, csv_out, audios_path):
    if not os.path.isfile(csv_out):
        extractor.add_hnr(subfolders, csv_in, csv_out, audios_path)

def main():
    extractor_opensmile = feature_extractor()
    extractor_praat     = praat_extractor()    

    # FraLusoPark
    extract_opensmile(extractor_opensmile, "fralusopark", paths.FRALUSOPARK_OUTPUT, "mfcc"  , ["CONTROLOS", "DOENTES"], paths.FRALUSOPARK_AUDIOS)
    extract_opensmile(extractor_opensmile, "fralusopark", paths.FRALUSOPARK_OUTPUT, "gemaps", ["CONTROLOS", "DOENTES"], paths.FRALUSOPARK_AUDIOS)
    extract_praat(
            extractor_praat,
            {
                0: "CONTROLOS",
                1: "DOENTES"
            },
            "{}/{}_{}_avg.csv".format(paths.FRALUSOPARK_OUTPUT, "fralusopark", "mfcc"), 
            "{}/{}_{}_avg.csv".format(paths.FRALUSOPARK_OUTPUT, "fralusopark", "praat"),
            paths.FRALUSOPARK_AUDIOS
    )

    # MDVR_KCL
    extract_opensmile(extractor_opensmile, "mdvr_kcl", paths.MDVR_KCL_OUTPUT, "mfcc", ["HC", "PD"], paths.MDVR_KCL_AUDIOS)

    # GITA 
    extract_opensmile(extractor_opensmile, "gita", paths.GITA_OUTPUT, "mfcc", ["hc", "pd"], paths.GITA_AUDIOS)



if __name__ == '__main__':
    main()
