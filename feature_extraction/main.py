import os
from multiprocessing import Process

from feature_extractor import feature_extractor
from praat_extractor   import praat_extractor

import paths

def extract_opensmile(extractor, dataset, output_path, feature_group, subfolders, audios_path, columns_to_use=None, of=None):
    output_file = "{}/{}_{}.csv".format(output_path, dataset, feature_group) if of is None else of
    if not os.path.isfile(output_file):
        pd_csv = extractor.extract_features(feature_group, "{}/{}".format(audios_path, subfolders[1]), output_path, False, columns_to_use=columns_to_use, audio_file='/afs/inesc-id.pt/corpora/fralusopark/Controlos/331/Marantz\ 2/331_G4_na_text.WAV')
        #hc_csv = extractor.extract_features(feature_group, "{}/{}".format(audios_path, subfolders[0]), output_path, True, columns_to_use=columns_to_use)
        
        extractor.concat_csv([pd_csv], output_file)
        #extractor.concat_csv([hc_csv, pd_csv], output_file)
        print ("{}/{}_{}.csv: extraction complete".format(output_path, dataset, feature_group))

    else:
        print("{}/{}_{}.csv already exists, ignoring".format(output_path, dataset, feature_group))
    
    return output_file

def extract_praat(extractor, subfolders, csv_in, csv_out, audios_path):
    #if not os.path.isfile(csv_out):
    extractor.add_hnr(subfolders, csv_in, csv_out, audios_path)
    #else:
    #    print("{} already exists, ignoring".format(csv_out))
    return csv_out

def run_one_dataset(extractor_opensmile, extractor_praat, dataset, dataset_output, dataset_hc_pd, audios_path):
    print("Process {} is running {}".format(os.getpid(), dataset))

    gemaps_cols = ['name', 'frameTime', 'jitterLocal_sma3nz', 'shimmerLocaldB_sma3nz']
    prosody_cols = ['name', 'frameTime', 'HNR_sma', 'F0_sma']

    mfcc    = extract_opensmile(extractor_opensmile, dataset, dataset_output, "mfcc"  , dataset_hc_pd, audios_path, of="mfcc.csv")
    #plp     = extract_opensmile(extractor_opensmile, dataset, dataset_output, "plp", dataset_hc_pd, audios_path, of="plp.csv")
    #prosody = extract_opensmile(extractor_opensmile, dataset, dataset_output, "prosody", dataset_hc_pd, audios_path, columns_to_use=prosody_cols)
    #gemaps  = extract_opensmile(extractor_opensmile, dataset, dataset_output, "gemaps", dataset_hc_pd, audios_path, columns_to_use=gemaps_cols, of='gemaps.csv')

    extractor_opensmile.merge([mfcc, plp, prosody, gemaps], ['name', 'frameTime'], "{}/{}_complete.csv".format(dataset_output, dataset))

def main():
    extractor_opensmile = feature_extractor()

    run_one_dataset(extractor_opensmile, None, 'gita', '', [0,1], None)
    extractor_opensmile.merge([gemaps], ['name', 'frameTime'], "patient_complete.csv")

def main2():
    extractor_opensmile = feature_extractor()
    extractor_praat     = praat_extractor()
    
    p_fralusopark = Process(target=run_one_dataset, args=(extractor_opensmile, extractor_praat, "fralusopark", paths.FRALUSOPARK_OUTPUT, ["CONTROLOS", "DOENTES"], paths.FRALUSOPARK_AUDIOS, ))
    p_fralusopark.start()

    p_gita = Process(target=run_one_dataset, args=(extractor_opensmile, extractor_praat, "gita", paths.GITA_OUTPUT, ["hc", "pd"], paths.GITA_AUDIOS, ))
    p_gita.start()

    p_mdvr_kcl = Process(target=run_one_dataset, args=(extractor_opensmile, extractor_praat, "mdvr_kcl", paths.MDVR_KCL_OUTPUT, ["HC", "PD"], paths.MDVR_KCL_AUDIOS, ))
    p_mdvr_kcl.start()

    p_fralusopark.join()
    p_gita.join()
    p_mdvr_kcl.join()

def main3():
    extractor_opensmile = feature_extractor()
    extractor_praat     = praat_extractor()    

    ###############
    # FraLusoPark #
    ###############
    #extract_opensmile(extractor_opensmile, "fralusopark", paths.FRALUSOPARK_OUTPUT, "gemaps", ["CONTROLOS", "DOENTES"], paths.FRALUSOPARK_AUDIOS)
    #mfcc  = extract_opensmile(extractor_opensmile, "fralusopark", paths.FRALUSOPARK_OUTPUT, "mfcc"  , ["CONTROLOS", "DOENTES"], paths.FRALUSOPARK_AUDIOS)
    #plp   = extract_opensmile(extractor_opensmile, "fralusopark", paths.FRALUSOPARK_OUTPUT, "plp", ["CONTROLOS", "DOENTES"], paths.FRALUSOPARK_AUDIOS)
    #extractor_opensmile.merge(mfcc, plp, ['name', 'frameTime'], "{}/{}_mfcc_plp.csv".format(paths.FRALUSOPARK_OUTPUT, "fralusopark"))
    #praat = extract_praat(
    #        extractor_praat,
    #        {
    #            0: "CONTROLOS",
    #            1: "DOENTES"
    #        },
    #        "{}/{}_{}_avg.csv".format(paths.FRALUSOPARK_OUTPUT, "fralusopark", "mfcc"), 
    #        "{}/{}_{}_avg.csv".format(paths.FRALUSOPARK_OUTPUT, "fralusopark", "praat"),
    #        paths.FRALUSOPARK_AUDIOS
    #)

    # Merge PRATT (which contains MFCC already) with PLP
    #extractor_opensmile.concat_csv([plp, praat], "{}/{}.csv".format(paths.FRALUSOPARK_OUTPUT, "fralusopark"))

    ############
    # MDVR_KCL #
    ############
    #mfcc  = extract_opensmile(extractor_opensmile, "mdvr_kcl", paths.MDVR_KCL_OUTPUT, "mfcc", ["HC", "PD"], paths.MDVR_KCL_AUDIOS)
    #plp   = extract_opensmile(extractor_opensmile, "mdvr_kcl", paths.MDVR_KCL_OUTPUT, "plp" , ["HC", "PD"], paths.MDVR_KCL_AUDIOS)
    #extractor_opensmile.merge(mfcc, plp, ['name', 'frameTime'], "{}/{}_mfcc_plp.csv".format(paths.MDVR_KCL_OUTPUT, "mdvr_kcl"))
    #praat = extract_praat(
    #        extractor_praat,
    #        {
    #            0: "HC",
    #            1: "PD"
    #        },
    #        "{}/{}_{}_avg.csv".format(paths.MDVR_KCL_OUTPUT, "mdvr_kcl", "mfcc"), 
    #        "{}/{}_{}_avg.csv".format(paths.MDVR_KCL_OUTPUT, "mdvr_kcl", "praat"),
    #        paths.MDVR_KCL_AUDIOS
    #)

    # Merge PRATT (which contains MFCC already) with PLP
    #extractor_opensmile.concat_csv([plp, praat], "{}/{}.csv".format(paths.MDVR_KCL_OUTPUT, "mdvr_kcl"))

    ########
    # GITA #
    ########
    #mfcc  = extract_opensmile(extractor_opensmile, "gita", paths.GITA_OUTPUT, "mfcc", ["hc", "pd"], paths.GITA_AUDIOS)
    #plp   = extract_opensmile(extractor_opensmile, "gita", paths.GITA_OUTPUT, "plp" , ["hc", "pd"], paths.GITA_AUDIOS)
    #extractor_opensmile.merge(mfcc, plp, ['name', 'frameTime'], "{}/{}_mfcc_plp.csv".format(paths.GITA_OUTPUT, 'gita'))
    #praat = extract_praat(
    #        extractor_praat,
    #        {
    #            0: "hc",
    #            1: "pd"
    #        },
    #        "{}/{}_{}_avg.csv".format(paths.GITA_OUTPUT, "gita", "mfcc"), 
    #        "{}/{}_{}_avg.csv".format(paths.GITA_OUTPUT, "gita", "praat"),
    #        paths.GITA_AUDIOS
    #)

    # Merge PRATT (which contains MFCC already) with PLP
    #extractor_opensmile.concat_csv([plp, praat], "{}/{}.csv".format(paths.GITA_OUTPUT, "gita"))


if __name__ == '__main__':
    main()
