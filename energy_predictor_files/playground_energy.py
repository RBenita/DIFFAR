
import splitfolders
import numpy as np
import json
import torchaudio
from tqdm import tqdm
import os
import csv
from pathlib import Path
import textgrid
from energy_predictor_utils import get_phonemes_from_file

def calc_energy_of_waveform_by_phonemes_durations(audio,duration_list,sr=16000):
    print("calc_energy")
    energy = []
    running_samples=0
    cur_duration_list = duration_list.split(",")
    cur_duration_list = list(map(float, cur_duration_list))
    for i in range(len(cur_duration_list)):
        if(cur_duration_list[i]==0):
             energy.append(cur_duration_list[i])
             print("devided by zero") 
             continue
        start = running_samples
        cur_dur = int(cur_duration_list[i]*sr)
        end = running_samples +cur_dur

        ## values not powered ##
        cur_phoneme_energy = np.sqrt((1/cur_dur)*sum(abs(audio[start:end])))

        # values powered by 2##
        # cur_phoneme_energy = np.sqrt((1/cur_dur)*sum(abs(audio[start:end]**2)))
        # print(f"start: {start/16000},    end:{end/16000},    dur:{cur_dur/16000}, enrgy:{cur_phoneme_energy}")
        running_samples = running_samples + cur_dur
        energy.append(cur_phoneme_energy) 
    energy = np.asarray(energy)
    return energy

##TODO##
def from_wav_file_to_npy_energy_file(wav_path, textgrid_path, npy_target_path):
    exts=[".wav"]
    for root, folders, files in os.walk(wav_path, followlinks=True):
        for file in files:
            file = Path(root) / file
            if file.suffix.lower() in exts:
                file_name = os.path.basename(file)
                file_name = os.path.splitext(file_name)[0]
                file_target_path = npy_target_path+ "/Energy_" + file_name +".npy"
                textgrid_file_path = textgrid_path  + "/" + file_name +".TextGrid"
                current_phonemes = textgrid.TextGrid.fromFile(textgrid_file_path)[1]
                Phonemes_list, Duration_list = get_phonemes_from_file(textgrid_file_path, current_phonemes)
                cur_audio, sr = torchaudio.load(file)
                cur_energy = calc_energy_of_waveform_by_phonemes_durations(cur_audio[0,:],Duration_list)                                                  
                np.save(file_target_path, cur_energy)

##TODO##
def making_identity_between_folders(path_a, path_b, origin_path,ext_a=".txt", ext_b=".wav",):
    len_ext_a = len(ext_a)
    count=0
    for filename in os.listdir(path_a):
        if filename[-len_ext_a:]==ext_a:
            wanted_file= os.path.join(origin_path,filename[:-len_ext_a]+ext_b)
            if os.path.isfile(wanted_file):
                count+=1
                print(wanted_file)
                os.system(f"cp {wanted_file} {path_b}")
            else :
                print(wanted_file)
                print(filename)
                print("it is not appeard")
                break
        else:
            print(f"it is not ext_a file: {filename}")
    print(count)


if __name__ == "__main__":

    ## Preparing the Energy npy file:
    # validation path:
    # file_name = "val_wavs_energy.npy"
    # wav_json_path = "/home/data/rbenita/LJSpeech_Align/LJSpeech-1.1/Duration_and_Energy_predictor_Dataset/wav_identity_splitted/val/val_wav.json"
    
    # # train path:
    # file_name = "test_wavs_energy_no_powered_with_file_name_2.npy"
    # csv_path = "/home/mlspeech/rbenita/PycharmProjects/energy_predictor_by_phonemes/csv_wavs_phonemes_duration/wav_files_phonemes_Duration_trainset.csv"
    
    # npy_target_path = f"/home/mlspeech/rbenita/PycharmProjects/energy_predictor_by_phonemes/energy_npy_files/{file_name}"
    # from_csv_to_npy_energy(csv_path, npy_target_path)
 
    # energy_array = np.load(npy_target_path, allow_pickle=True)
    # print(len(energy_array))
    # print(energy_array[0:5])

    ##making a copy of files just with different extension:
    # dataset="train"
    path_a=f"/home/data/rbenita/data_for_test_results/65_attempts/original_text_files"
    path_b=f"/home/data/rbenita/data_for_test_results/65_attempts/original_wav_files_16000"
    origin_path = f"/home/data/rbenita/data_for_test_results/backup_test_wav_and_texts_original"
    making_identity_between_folders(path_a, path_b, origin_path)

    # train path:
    # cur_wav_path = "/home/data/rbenita/CSTR_VCTK_original_dataset/all_data_mic_2_16000_less_sielence_splitted/train/all_data"
    # cur_textgrid_path="/home/data/rbenita/CSTR_VCTK_original_dataset/mic_2_aligend_split/train/all_data"
    # npy_target_path = f"/home/data/rbenita/CSTR_VCTK_original_dataset/Conditioned_Energy_Splitted_No_Power/train/all_data"
    
    # from_wav_file_to_npy_energy_file(cur_wav_path,cur_textgrid_path, npy_target_path)
 

 