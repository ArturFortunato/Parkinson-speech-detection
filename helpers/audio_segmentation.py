import os
from pydub import AudioSegment
from pathlib import Path

MDVK_KCL_WAV_PATH      = '../../datasets/MDVR_KCL/26-29_09_2017_KCL/ReadText/'
MDVR_KCL_WAV_CLEAN     = '../../mdvr_kcl_wav/'
MDVR_KCL_SEGMENTS_FILE = 'mdvr_kcl_segments.txt'

def get_all_audios_segments(segments_file):
    segments = {}
    current_file = ''

    segments_file = open(segments_file, 'r')
    lines = segments_file.readlines()
	
    for line in lines:
        # New audio file
        if line.startswith("./"):
            # :-1 removes \n at the end of the line
            segments[line[:-1]] = []
            current_file = line[:-1]

        else:
            [init, end] = line[:-1].split("-")
            segments[current_file].append((init, end))

    return segments

def write_audio_segments_to_file(filename, audio):
    audio.export(MDVR_KCL_WAV_CLEAN + filename, format="wav")

def segment_audios(segments, output_dir):
    for filename in segments.keys():
        print("Segmenting {}".format(filename))
        audio = AudioSegment.from_wav(MDVK_KCL_WAV_PATH  + filename)
        
        result_audio = None
        for segment in segments[filename]:
            init = int(segment[0]) * 1000 # To Milliseconds
            end  = int(segment[1]) * 1000 # To Milliseconds
            
            if result_audio is None:
                result_audio = audio[init:end]
            else:
                result_audio = result_audio.append(audio[init:end])
        
        write_audio_segments_to_file(filename, result_audio)

def main():
    segments = get_all_audios_segments(MDVR_KCL_SEGMENTS_FILE)
    
    segment_audios(segments, MDVR_KCL_WAV_CLEAN)
	
if __name__ == '__main__':
    main()
