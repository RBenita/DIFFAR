import numpy as np
import os
import torch
import torchaudio
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import json
import random
from pathlib import Path
import textgrid
from model import DiffAR
from utils_for_inference import Build_excisting_phonemes_sec_approach, get_overlap_duration
import time
from params import AttrDict, params as base_params
from duration_predictors_files.Text_To_Speech_Procedure import producing_filename_text_phoneme_list, test_on_given_lists
from energy_predictor_files.Text_To_Speech_Procedure import producing_phoneme_list, test_on_given_lists_energy
models = {}

def plot_waveform(waveform, sample_rate, title="Waveform", path="path", save_fig=False , xlim=None, ylim=None):
    waveform = waveform.numpy()
    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate
    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].plot(time_axis, waveform[c], linewidth=1)
        axes[c].grid(True)
        if num_channels > 1:
            axes[c].set_ylabel(f'Channel {c + 1}')
        if xlim:
            axes[c].set_xlim(xlim)
        if ylim:
            axes[c].set_ylim(ylim)
    figure.suptitle(title)
    if save_fig:
        plt.savefig(path)

def build_embedding_table(max_steps):
        steps = torch.arange(max_steps).unsqueeze(1)    # [T,1]
        dims = torch.arange(64).unsqueeze(0)                    # [1,64]
        table = steps * 10.0**(dims * 4.0 / 63.0)         # [T,64]
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)
        return table

def load_model(model_dir=None, params=None, device=torch.device('cuda',6)):
    if not model_dir in models:
        if os.path.exists(f'{model_dir}/weights.pt'):
            checkpoint = torch.load(f'{model_dir}/weights.pt')
        else:
            checkpoint = torch.load(model_dir)
        model = DiffAR(AttrDict(base_params)).to(device)
        model.load_state_dict(checkpoint['model'])
        model.eval()
        models[model_dir] = model
        print("DiffAR model loaded")

    model = models[model_dir]
    model.params.override(params)
    return model

def inference_process(conditioned_window, conditioned_phonemes_sig, conditioned_energy_sig, cur_model=None, params=None, device=torch.device('cuda',6)):

    with torch.no_grad():
        beta = np.array(cur_model.params.noise_schedule)
        alpha = 1 - beta
        alpha_cum = np.cumprod(alpha)
        audio = torch.randn(conditioned_window.shape[0], conditioned_window.shape[1], device=device)
        assert audio.shape == conditioned_window.shape and conditioned_window.shape == conditioned_phonemes_sig.shape, f"{audio.shape}, {conditioned_window.shape}, {conditioned_phonemes_sig.shape} "
        print(f"Current frame size: {audio.shape}")
        for n in range(len(alpha) - 1, -1, -1):
            c1 = 1 / alpha[n]**0.5
            c2 = beta[n] / (1 - alpha_cum[n])**0.5
            ## there is no conditiner for now, and the model isnt using this input
            audio = c1 * (audio.to(device) - c2 * cur_model(audio, conditioned_window.to(device), torch.tensor([n], device=audio.device), conditioned_phonemes_sig.to(device), conditioned_energy_sig.to(device)).squeeze(1))
            if n > 0:
                noise = torch.randn_like(audio)
                sigma = ((1.0 - alpha_cum[n-1]) / (1.0 - alpha_cum[n]) * beta[n])**0.5
                audio += sigma * noise
            device = torch.device('cuda',6)
    return audio, cur_model.params.sample_rate

def find_the_next_window_borders(phonemes, start_left_phonemes, end_tot, window_length, finished, partial_phoneme_flag):
    start_cur_win = start_left_phonemes
    end_cur_win = start_left_phonemes
    current_total_length_window = 0
    starting_with_partial = False
    
    for phoneme in phonemes:
        phoneme_max =  round(phoneme.maxTime * base_params.sample_rate)
        phoneme_min =  round(phoneme.minTime * base_params.sample_rate)
   
        if (partial_phoneme_flag==True) and (phoneme_min<end_cur_win) and (end_cur_win<phoneme_max):
            ## That mean we found the partial phoneme, we need to take just the added length - the end of the last phoneme from the last window
            second_phoneme_part = phoneme_max - end_cur_win
            print(f"partial_phoneme_flag= {partial_phoneme_flag}")
            assert current_total_length_window + second_phoneme_part == base_params.window_length/2, f"somehow the partial part from last window isnt haldf window and it is {current_total_length_window + second_phoneme_part} {phoneme_min} {end_cur_win}  {phoneme_max}"            
            current_total_length_window += second_phoneme_part
            end_cur_win = phoneme_max
            partial_phoneme_flag=False
            starting_with_partial=True

        if end_tot < phoneme_max:
            finished = True
            break

        if end_cur_win == phoneme_min:
            phoneme_length = phoneme_max - phoneme_min
            if current_total_length_window + phoneme_length <= window_length:
                current_total_length_window += phoneme_length
                end_cur_win = phoneme_max
                continue
            
            else:
                ## in case the first full phoneme is big and didnt add - force adding it for continuity/
                if starting_with_partial and current_total_length_window==base_params.window_length/2:
                    current_total_length_window += phoneme_length
                    end_cur_win = phoneme_max
                    # print("two long phonemes")
                    break
                if current_total_length_window<=base_params.window_length/2:
                    current_total_length_window += phoneme_length
                    end_cur_win = phoneme_max
                    # print("I think that it is not good realy small because no continuity - maybe in 150ms will be problematic")
                    # print("thereforme maybe the next one will be very long")
                    break

                if np.abs(current_total_length_window + phoneme_length-window_length) <= np.abs(current_total_length_window - window_length):
                    current_total_length_window += phoneme_length
                    end_cur_win = phoneme_max
                    break
                else:
                    break

    assert not(round(end_cur_win)==round(start_cur_win)) , f"Didnt add time to the cureent phoneme. is all the values rounded? end_cur_win: {end_cur_win}, start_cur_win: {start_cur_win}, phoneme_max: {phoneme_max}, phoneme_min: {phoneme_min} "
    
    ## make it finished if he got to the end and this is apparently the last frame.
    if (current_total_length_window < base_params.window_length/2):
        print("Finishing")
        finished = True

        # return start_cur_win, end_cur_win, current_total_length_window, finished
    return int(round(end_cur_win)), finished

def get_overlap_duration_from_end(start_cur_phonemes, end_cur_phonemes, info_taken_phonemes, window_length):
    partial_phoneme_flag_decision = False
    total_phonemes = len(info_taken_phonemes)
    assert total_phonemes>=1, "There must be at list one phoneme in a current window"
    desirable_phonemes = np.abs((-total_phonemes)//3)
    assert desirable_phonemes>=1, "There must be at list one phoneme in the overlap area"
    assert end_cur_phonemes == round(info_taken_phonemes[-1].maxTime*base_params.sample_rate), f"The last phoneme end should be the end of the windows. end_cur_phonemes: {end_cur_phonemes}, {info_taken_phonemes[-1].maxTime*base_params.sample_rate}  "
    overlap_duration = 0
    i=1
    while desirable_phonemes-i>=0:
        if end_cur_phonemes - info_taken_phonemes[-i].minTime*base_params.sample_rate  <=  int(base_params.window_length/2):
            overlap_duration = end_cur_phonemes - info_taken_phonemes[-i].minTime*base_params.sample_rate
            i+=1
        else:
            break

    if overlap_duration==0:
        overlap_duration = int(round(base_params.window_length/2))
        partial_phoneme_flag_decision = True
        print(window_length)
    
    return int(round(overlap_duration)), partial_phoneme_flag_decision

def fixing_duration_and_zone(overlap_zone_fun, overlap_duration_fun, conditioned_phonemes_signal_fun, conditioned_energy_signal_fun, list_info_cur_taken_phonemes_fun, total_length_curr_window_fun, start_cur_phonemes_fun, partial_phoneme_flag):
    ## In this function, only cases which overlap_zone<duration_predictor relevant.
    gap=0
    k=0
    if(partial_phoneme_flag):
        # In case a half from base_params.windows_size is taken (It is too much).
        # We want half from the current window:
        gap = round(overlap_duration_fun) - round(overlap_zone_fun)
        print(f"im in partial phoneme: overlap_duration_fun = {overlap_duration_fun}, overlap_zone_fun= {overlap_zone_fun}, gap={gap}")
        conditioned_phonemes_signal_fun = conditioned_phonemes_signal_fun[:,gap:]
        conditioned_energy_signal_fun = conditioned_energy_signal_fun[:,gap:]
        total_length_curr_window_fun = total_length_curr_window_fun - gap
        start_cur_phonemes_fun = start_cur_phonemes_fun + gap
        overlap_duration_fun -= gap

        return overlap_duration_fun, start_cur_phonemes_fun, total_length_curr_window_fun, conditioned_phonemes_signal_fun, conditioned_energy_signal_fun, list_info_cur_taken_phonemes_fun 
    
    else:
        will_be_removed=0
        while(overlap_duration_fun-will_be_removed>overlap_zone_fun):
            ##actually it is supposed to end when they are equal because on this time the phoneme are complete
            current_phoneme_min = round(list_info_cur_taken_phonemes_fun[k].minTime*16000)
            current_phoneme_max = round(list_info_cur_taken_phonemes_fun[k].maxTime*16000)
            current_phoneme_duration = current_phoneme_max - current_phoneme_min
            will_be_removed += current_phoneme_duration
            k+=1

        if overlap_duration_fun-will_be_removed==0:
            print("Too much deleted")
            k-=1
            will_be_removed -= current_phoneme_duration

        conditioned_phonemes_signal_fun = conditioned_phonemes_signal_fun[:,will_be_removed:]
        conditioned_energy_signal_fun = conditioned_energy_signal_fun[:,will_be_removed:]
        total_length_curr_window_fun = total_length_curr_window_fun - will_be_removed   
        list_info_cur_taken_phonemes_fun =  list_info_cur_taken_phonemes_fun[k:]
        start_cur_phonemes_fun = start_cur_phonemes_fun + will_be_removed
        overlap_duration_fun -= will_be_removed
        return overlap_duration_fun, start_cur_phonemes_fun, total_length_curr_window_fun, conditioned_phonemes_signal_fun, conditioned_energy_signal_fun, list_info_cur_taken_phonemes_fun 

def predict_cond_model_sec_approach_list_full_wav(Textgrid_path, Energy_npy_path, cur_model, sampling_rate=16000):
    partial_phoneme = False
    print("Started loading information")

    audio_a, sr = torchaudio.load(f"/home/mlspeech/rbenita/PycharmProjects/git_models/DiffAR_200/empty_sec_audio.wav")
    phoneme = textgrid.TextGrid.fromFile(Textgrid_path)[1]
    Energy = np.load(Energy_npy_path)
    if not(len(Energy)==len(phoneme)):
        print("Energy padding first None Phoneme")
        Energy = np.insert(Energy, 0, 0.03)
    start_total_audio = 0
    end_total_audio = round(phoneme.maxTime * sampling_rate)
    print(f"The toatal length by the predicted phonemes is: {end_total_audio}")
    initial_frame = True
    finished = False    
    start_cur_phonemes = start_total_audio
    i=1
    print("Finished getting information")

    while not(finished):
        end_cur_phonemes, finished= find_the_next_window_borders(phoneme,start_cur_phonemes,end_total_audio,base_params.window_length,finished, partial_phoneme)
        if(initial_frame):
            total_length_curr_window = int(round(end_cur_phonemes-start_cur_phonemes))
            conditioned_audio = torch.zeros((1,total_length_curr_window))
            conditioned_phonemes_signal, list_info_cur_taken_phonemes, conditioned_energy_signal = \
            Build_excisting_phonemes_sec_approach(start_cur_phonemes, end_cur_phonemes ,phoneme,Energy, base_params.sample_rate)

            print(f"Current phonemes: {list_info_cur_taken_phonemes}")
            overlap_zone = int(get_overlap_duration(start_cur_phonemes, list_info_cur_taken_phonemes, total_length_curr_window))
            conditioned_audio[:,:] = audio_a[:,:total_length_curr_window]
            conditioned_audio[:,overlap_zone:]=0
            generated_final_audio, sr = inference_process(conditioned_audio, conditioned_phonemes_signal, conditioned_energy_signal, cur_model)
            initial_frame = False
        else:
            total_length_curr_window = int(round(end_cur_phonemes-start_cur_phonemes))
            conditioned_audio = torch.zeros((1,total_length_curr_window))
            conditioned_phonemes_signal, list_info_cur_taken_phonemes, conditioned_energy_signal = \
            Build_excisting_phonemes_sec_approach(start_cur_phonemes, end_cur_phonemes ,phoneme,Energy, base_params.sample_rate)

            overlap_zone = int(get_overlap_duration(start_cur_phonemes, list_info_cur_taken_phonemes, total_length_curr_window))
            assert not(total_length_curr_window==0), f"Total length is zero, overlap_duration:  {overlap_duration} , overlap_zone {overlap_zone},start_cur_phonemes: {start_cur_phonemes}, end_cur_phonemes {end_cur_phonemes}, end_total_audio: {end_total_audio} "

            if(overlap_zone < overlap_duration):
                overlap_duration, start_cur_phonemes, total_length_curr_window, conditioned_phonemes_signal, conditioned_energy_signal,  list_info_cur_taken_phonemes = \
                    fixing_duration_and_zone(overlap_zone, overlap_duration, conditioned_phonemes_signal,conditioned_energy_signal, list_info_cur_taken_phonemes, total_length_curr_window, start_cur_phonemes, partial_phoneme)
                conditioned_audio = torch.zeros((1,total_length_curr_window))

            print(f"Current phonemes: {list_info_cur_taken_phonemes}")

            
            #TODO: Write a comprehansive explanation.
            # In case the current overlap needs more then the phonemes should be here - we need to prevent it.
            # The maximum it will get is the overlap from the end of the last frame
            conditioned_audio[:,:overlap_duration] = generated_final_audio[:,-overlap_duration:]
            conditioned_audio[:,overlap_duration:]=0
            generated_current_audio, sr = inference_process(conditioned_audio, conditioned_phonemes_signal, conditioned_energy_signal, cur_model)
            generated_final_audio= torch.cat((generated_final_audio[:,:],generated_current_audio[:,overlap_duration:]),-1)
            i+=1

        ### Preparing the next prediction:
        overlap_duration, partial_phoneme= get_overlap_duration_from_end(start_cur_phonemes, end_cur_phonemes, list_info_cur_taken_phonemes, total_length_curr_window)
        start_cur_phonemes = end_cur_phonemes - overlap_duration

    return audio_a, generated_final_audio, sr

def open_folder(main_path, folder_name):
    try:
        # Attempt to create the directory
        folder_path = os.path.join(main_path,folder_name)
        os.mkdir(folder_path)
        print(f"Directory '{folder_path}' created successfully.")
    except OSError as e:
            print(f"Directory '{folder_path}' already exists.")
            print(f"or Error creating directory '{folder_path}': {e}")
    print(folder_path)
    return folder_path

def main(args):
    print(args)
    if args.main_directory==None:
    # Specify the directory path you want to create
        directory_path = '/home/mlspeech/rbenita/PycharmProjects/git_models/DiffAR_200/'
    else:
        directory_path = args.main_directory

    
    ### from text files predict textgridfiles ###
    origin_text_path_ = os.path.join(directory_path,"text_files")

    ### using G2P MODEL
    print("Start predicting Textgrids")
    path_target_predicted_textgrid = open_folder(directory_path, "predicted_TextGrid_files")
    file_names_and_text_list = producing_filename_text_phoneme_list(origin_text_path_, None)
    Is_G2P = True
    duration_model_path = os.path.join(os.getcwd(),"duration_predictors_files/saved_models/predictor_model_sec_dividing_kernel_5.ckpt") 
    test_on_given_lists(file_names_and_text_list,path_target_predicted_textgrid,duration_model_path,Is_G2P)
    print("Finish predicting Textgrids")
    
### from textgrid files predict energy files ###
    print("Start predicting Energies")

    ### using phonemse from G2P MODEL
    path_target_predicted_energy = open_folder(directory_path, "predicted_energy_files")
    file_names_and_phoneme_list = producing_phoneme_list(path_target_predicted_textgrid)
    energy_model_path = os.path.join(os.getcwd(),"energy_predictor_files/saved_models/energy_predictor_model_2_layers_7_5_k.ckpt") 
    test_on_given_lists_energy(file_names_and_phoneme_list,path_target_predicted_energy, energy_model_path)
    print("Finish predicting Energies")

### predict several wavs from a folder: 
    current_model =  os.path.join(os.getcwd(),"models/DiffAR_200.pt")
    target_path = open_folder(directory_path, "generated_wavs")
    Textgrid_path = path_target_predicted_textgrid
    Npy_path = path_target_predicted_energy

    cur_model_ = load_model(current_model)
    iter_per_file = 1
    enable_seed = True

    ext = ('.TextGrid')
    for cur_iter in range(iter_per_file):
        print(f"cur_iter= {cur_iter}")
        for files in os.listdir(Textgrid_path):
            if enable_seed:
                random_seed = cur_iter + 1
                print(f"current random seed: {random_seed}")
                torch.manual_seed(random_seed)
            if files.endswith(ext):
                print(f"target path: {target_path}")
                file_name = os.path.basename(files)
                file_name = os.path.splitext(file_name)[0]
                cur_TextGrid_path = f"{Textgrid_path}/{files}"
                cur_Energy_npy_path =  f"{Npy_path}/predicted_Energy_predicted_{files[10:-9]}" + ".npy"
                t0= time.time()
                original_audio, generated_final_audio, sr = predict_cond_model_sec_approach_list_full_wav(cur_TextGrid_path,cur_Energy_npy_path,cur_model_)
                t1 = time.time() - t0
                torchaudio.save(f"{target_path}/generated_{cur_iter}_{file_name}.wav" ,
                        generated_final_audio.cpu(), sample_rate=16000)

if __name__ == '__main__':
     parser = ArgumentParser(description='TTS procedure OF DiffAR model')
     parser.add_argument('--main_directory', default = None,
             help='directory in which all files and folders are stored')
     main(parser.parse_args())

    