import os

from feature_extractor import feature_extractor
from praat_extractor   import praat_extractor

import paths

def extract_opensmile(extractor, dataset, output_path, feature_group, subfolders, audios_path):
    if not os.path.isfile("{}/{}_{}.csv".format(output_path, dataset, feature_group)):
        hc_csv = extractor.extract_features(feature_group, "{}/{}".format(audios_path, subfolders[0]), output_path, True )
        pd_csv = extractor.extract_features(feature_group, "{}/{}".format(audios_path, subfolders[1]), output_path, False)

        extractor.merge_csv([hc_csv, pd_csv], "{}/{}_{}.csv".format(output_path, dataset, feature_group))
        print ("{}/{}_{}.csv: extraction complete".format(output_path, dataset, feature_group))

    else:
        print("{}/{}_{}.csv already exists, ignoring".format(output_path, dataset, feature_group))

    averaged = "{}/{}_{}_avg.csv".format(output_path, dataset, feature_group)
    if not os.path.isfile(averaged):
        extractor.average_and_write_csv("{}/{}_{}.csv".format(output_path, dataset, feature_group), averaged)
        print("{}: extraction complete".format(averaged))

    else:
        print("{} already exists, ignoring".format(averaged))
    
    return averaged

def extract_praat(extractor, subfolders, csv_in, csv_out, audios_path):
    if not os.path.isfile(csv_out):
        extractor.add_hnr(subfolders, csv_in, csv_out, audios_path)
    else:
        print("{} already exists, ignoring".format(csv_out))
    return csv_out

def main():
    extractor_opensmile = feature_extractor()
    extractor_praat     = praat_extractor()    

    ###############
    # FraLusoPark #
    ###############
    #extract_opensmile(extractor_opensmile, "fralusopark", paths.FRALUSOPARK_OUTPUT, "gemaps", ["CONTROLOS", "DOENTES"], paths.FRALUSOPARK_AUDIOS)
    mfcc  = extract_opensmile(extractor_opensmile, "fralusopark", paths.FRALUSOPARK_OUTPUT, "mfcc"  , ["CONTROLOS", "DOENTES"], paths.FRALUSOPARK_AUDIOS)
    plp   = extract_opensmile(extractor_opensmile, "fralusopark", paths.FRALUSOPARK_OUTPUT, "plp", ["CONTROLOS", "DOENTES"], paths.FRALUSOPARK_AUDIOS)
    praat = extract_praat(
            extractor_praat,
            {
                0: "CONTROLOS",
                1: "DOENTES"
            },
            "{}/{}_{}_avg.csv".format(paths.FRALUSOPARK_OUTPUT, "fralusopark", "mfcc"), 
            "{}/{}_{}_avg.csv".format(paths.FRALUSOPARK_OUTPUT, "fralusopark", "praat"),
            paths.FRALUSOPARK_AUDIOS
    )

    # Merge PRATT (which contains MFCC already) with PLP
    extractor_opensmile.merge_csv([plp, praat], "{}/{}.csv".format(paths.FRALUSOPARK_OUTPUT, "fralusopark"))

    ############
    # MDVR_KCL #
    ############
    mfcc  = extract_opensmile(extractor_opensmile, "mdvr_kcl", paths.MDVR_KCL_OUTPUT, "mfcc", ["HC", "PD"], paths.MDVR_KCL_AUDIOS)
    plp   = extract_opensmile(extractor_opensmile, "mdvr_kcl", paths.MDVR_KCL_OUTPUT, "plp" , ["HC", "PD"], paths.MDVR_KCL_AUDIOS)
    praat = extract_praat(
            extractor_praat,
            {
                0: "HC",
                1: "PD"
            },
            "{}/{}_{}_avg.csv".format(paths.MDVR_KCL_OUTPUT, "mdvr_kcl", "mfcc"), 
            "{}/{}_{}_avg.csv".format(paths.MDVR_KCL_OUTPUT, "mdvr_kcl", "praat"),
            paths.MDVR_KCL_AUDIOS
    )

    # Merge PRATT (which contains MFCC already) with PLP
    extractor_opensmile.merge_csv([plp, praat], "{}/{}.csv".format(paths.MDVR_KCL_OUTPUT, "mdvr_kcl"))

    ########
    # GITA #
    ########
    mfcc  = extract_opensmile(extractor_opensmile, "gita", paths.GITA_OUTPUT, "mfcc", ["hc", "pd"], paths.GITA_AUDIOS)
    plp   = extract_opensmile(extractor_opensmile, "gita", paths.GITA_OUTPUT, "plp" , ["hc", "pd"], paths.GITA_AUDIOS)
    praat = extract_praat(
            extractor_praat,
            {
                0: "hc",
                1: "pd"
            },
            "{}/{}_{}_avg.csv".format(paths.GITA_OUTPUT, "gita", "mfcc"), 
            "{}/{}_{}_avg.csv".format(paths.GITA_OUTPUT, "gita", "praat"),
            paths.GITA_AUDIOS
    )

    # Merge PRATT (which contains MFCC already) with PLP
    extractor_opensmile.merge_csv([plp, praat], "{}/{}.csv".format(paths.GITA_OUTPUT, "gita"))


if __name__ == '__main__':
    main()
