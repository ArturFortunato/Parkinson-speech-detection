import os

from feature_extractor import feature_extractor

FRALUSOPARK_AUDIOS = '/afs/inesc-id.pt/home/aof/thesis/fralusopark_wav'
FRALUSOPARK_OUTPUT = '/afs/inesc-id.pt/home/aof/thesis/features/fralusopark/'

MDVR_KCL_AUDIOS = '/afs/inesc-id.pt/home/aof/thesis/mdvr_kcl_wav/'
MDVR_KCL_OUTPUT = '/afs/inesc-id.pt/home/aof/thesis/features/mdvr_kcl/'

GITA_AUDIOS = '/afs/inesc-id.pt/corpora/pc-gita/PC-GITA_per_task_44100Hz/read_text/ayerfuialmedico/sin_normalizar/'
GITA_OUTPUT = '/afs/inesc-id.pt/home/aof/thesis/features/gita/'

def extract(extractor, dataset, output_path, feature_group, subfolders, audios_path):
    # FraLusoPark
    if not os.path.isfile("{}/{}_{}.csv".format(FRALUSOPARK_OUTPUT, dataset, feature_group)):
        hc_csv = extractor.extract_features(feature_group, "{}/{}".format(audios_path, subfolders[0]), output_path, True )
        pd_csv = extractor.extract_features(feature_group, "{}/{}".format(audios_path, subfolders[1]), output_path, False)

        extractor.merge_csv([hc_csv, pd_csv], "{}/{}_{}.csv".format(output_path, dataset, feature_group))



def main():
    extractor = feature_extractor()
    
    # FraLusoPark
    extract(extractor, "fralusopark", FRALUSOPARK_OUTPUT, "mfcc", ["CONTROLOS", "DOENTES"], FRALUSOPARK_AUDIOS)

    # MDVR_KCL
    extract(extractor, "mdvr_kcl", MDVR_KCL_OUTPUT, "mfcc", ["HC", "PD"], MDVR_KCL_AUDIOS)

    # GITA 
    extract(extractor, "gita", GITA_OUTPUT, "mfcc", ["hc", "pd"], GITA_AUDIOS)

if __name__ == '__main__':
    main()
