import wave
import contextlib
import os 

FRALUSOPARK_RAW_PATH = '/afs/inesc-id.pt/corpora/fralusopark/processed_data/manually_processed/'
FRALUSOPARK_WAV_PATH = '../../fralusopark_wav/'
GITA_WAV_PATH        = '/afs/inesc-id.pt/corpora/pc-gita/PC-GITA_per_task_44100Hz/read_text/ayerfuialmedico/sin_normalizar/'
MDVK_KCL_WAV_PATH    = '../../../datasets/MDVR_KCL/26-29_09_2017_KCL/ReadText/'

def get_files_path(folder, pattern):
    return [folder + "/" + f for f in os.listdir(folder) if f.endswith(pattern)]
    
def get_audio_duration(path):
    with contextlib.closing(wave.open(path,'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        duration = frames / float(rate)
    
    return duration 

def get_total_and_avg_audio_duration(files):
    duration = 0.0
    for f in files:
        duration += get_audio_duration(f)
    
    return duration, duration / len(files)

def convert_raw_files_to_wav(files):
    for f in files:
        path_splitted = [folder for folder in f.split("/") if folder != "raws"]
        raw_to_wav(f, "./wav/" + "/".join(path_splitted[-2:])[:-3] + "wav")

def raw_to_wav(raw, wav):
    with open(raw, "rb") as inp_f:
        data = inp_f.read()
        with wave.open(wav, "w") as out_f:
            out_f.setnchannels(1)
            out_f.setsampwidth(2) # number of bytes 2
            out_f.setframerate(16000)
            out_f.writeframesraw(data)

def has_converted(path):
    return len(os.listdir(path)) != 0 

def print_total_and_avg_duration(label, files):
    total, avg = get_total_and_avg_audio_duration(files)
    num_files = len(files)

    print("\n{} group: {} \nAudios total duration: {} minutes, {} seconds \nAudios average duration: {} minutes, {} seconds\n".format(label, num_files, total // 60, total % 60, avg // 60, avg % 60))

def main():
    print("============FRALUSOPARK==============\n\n")

    if not has_converted(FRALUSOPARK_WAV_PATH + 'DOENTES/'):
        pd_raw_files = get_files_path(FRALUSOPARK_RAW_PATH + 'DOENTES/raws', '_text.raw') 
        convert_raw_files_to_wav(pd_raw_files)

    if not has_converted(FRALUSOPARK_WAV_PATH + 'CONTROLOS/'):
        hc_raw_files = get_files_path(FRALUSOPARK_RAW_PATH + 'CONTROLOS/raws', '_text.raw') 
        convert_raw_files_to_wav(hc_raw_files)
    
    pd_wav_files = get_files_path(FRALUSOPARK_WAV_PATH + 'DOENTES/', '')
    hc_wav_files = get_files_path(FRALUSOPARK_WAV_PATH + 'CONTROLOS/', '')

    print_total_and_avg_duration("PD", pd_wav_files)
    print_total_and_avg_duration("HC", hc_wav_files)


    print("===============GITA==================\n\n")
    
    pd_wav_files = get_files_path(GITA_WAV_PATH + 'pd', '')
    hc_wav_files = get_files_path(GITA_WAV_PATH + 'hc', '')
    
    print_total_and_avg_duration("PD", pd_wav_files)
    print_total_and_avg_duration("HC", hc_wav_files)

    print("===============MDVR_KCL==================\n\n")
    
    
    pd_wav_files = get_files_path(MDVK_KCL_WAV_PATH + 'PD', '')
    hc_wav_files = get_files_path(MDVK_KCL_WAV_PATH + 'HC', '')
    
    print_total_and_avg_duration("PD", pd_wav_files)
    print_total_and_avg_duration("HC", hc_wav_files)

if __name__ == '__main__':
    main()
