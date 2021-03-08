from feature_extractor import feature_extractor

FRALUSOPARK_AUDIOS = '/afs/inesc-id.pt/home/aof/thesis/fralusopark_wav'
FRALUSOPARK_OUTPUT = '/afs/inesc-id.pt/home/aof/thesis/features/fralusopark/'

def main():
    extractor = feature_extractor()
    
    hc_csv = extractor.extract_features('mfcc', "{}/CONTROLOS".format(FRALUSOPARK_AUDIOS), FRALUSOPARK_OUTPUT, True )
    pd_csv = extractor.extract_features('mfcc', "{}/DOENTES".format(FRALUSOPARK_AUDIOS),   FRALUSOPARK_OUTPUT, False)

    extractor.merge_csv([hc_csv, pd_csv], "{}/fralusopark.csv".format(FRALUSOPARK_OUTPUT))

if __name__ == '__main__':
    main()
