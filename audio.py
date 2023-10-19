import random
import os
import sys
import json
from pathlib import Path
import logging
import textgrid
import torch
import torchaudio
import math
import numpy as np
torchaudio.set_audio_backend("sox_io")
from learner import logger

REPRESENT_LENGTH = 128
TOTAL_AVAILABLE_PHONEMES = 72

phoneme_to_index_dict = { "" : 1, "AA0": 2, 'AA1': 3, 'AA2': 4, 'AE0': 5 ,'AE1': 6
 ,'AE2': 7 ,'AH0': 8 ,'AH1': 9 ,'AH2': 10 ,'AO0': 11 ,'AO1': 12 ,'AO2': 13 ,'AW0': 14 ,'AW1': 15
   ,'AW2': 16 ,'AY0': 17 ,'AY1': 18 ,'AY2': 19,'B': 20 ,'CH': 21, 'D' : 22, 'DH' :23,
   'EH0':24, 'EH1':25, 'EH2':26 ,'ER0':27, 'ER1':28,'ER2':29 ,'EY0':30 , 'EY1':31 ,'EY2':32 ,'F':33 ,
   'G':34 ,'HH':35 ,'IH0':36 ,'IH1':37 ,'IH2':38 ,'IY0':39, 'IY1':40 , 'IY2':41 , 'JH':42,
    'K':43,  'L':44,  'M':45,  'N':46 , 'NG':47,  'OW0':48, 'OW1':49, 'OW2':50,'OY0':51, 'OY1':52, 
    'OY2':53, 'P':54, 'R':55, 'S':56, 'SH':57, 'T':58, 'TH':59, 'UH0':60, 'UH1':61,
     'UH2':62, 'UW0':63, 'UW1':64, 'UW2':65, 'V':66, 'W':67, 'Y':68, 'Z':69, 'ZH':70, 'spn':71, 'sil':72
}

##TODO##
def find_TextGrid_files(path, exts=[".textgrid"], progress=True):
    """
    dump all files in the given path to a json file with the format:
    [(file_path),...]
    """
    audio_files = []
    for root, folders, files in os.walk(path, followlinks=True):
        for file in files:
            file = Path(root) / file
            if file.suffix.lower() in exts:
                audio_files.append(str(file.resolve()))    
    meta = []
    for idx, file in enumerate(audio_files):
        meta.append((file))
        if progress:
            print(format((1 + idx) / len(audio_files), " 3.1%"), end='\r', file=sys.stderr)
    meta.sort()
    return meta

##TODO##
def find_audio_files(path, exts=[".wav"], progress=True):
    """
    dump all files in the given path to a json file with the format:
    [(audio_path, audio_length),...]
    """
    audio_files = []
    for root, folders, files in os.walk(path, followlinks=True):
        for file in files:
            file = Path(root) / file
            if file.suffix.lower() in exts:
                audio_files.append(str(file.resolve()))
    meta = []
    for idx, file in enumerate(audio_files):
        siginfo = torchaudio.info(file)
        length = siginfo.num_frames // siginfo.num_channels
        meta.append((file, length))
        if progress:
            print(format((1 + idx) / len(audio_files), " 3.1%"), end='\r', file=sys.stderr)
    meta.sort()
    return meta

def get_overlap_duration(start_p, info_taken_phonemes, window_length):
    ### In case of conditional synthesis, we would like to choose which phonemes 
    ### from the last window will apeear in the current window.
    ### we assume that aprox 1/3 phonemes from the last window is involving in the current window
    ### and also assume that this part will be 1/3 phonemes from the current frame.
    ### the function gets the info about the phonemes on a specific window
    ### the function return the overlap (which part of the window is visible)

    total_phonemes = len(info_taken_phonemes)

    assert total_phonemes>=1, "There must be at list one phoneme in a current window"
    desirable_phonemes = np.abs((-total_phonemes)//3)
    assert desirable_phonemes>=1, "There must be at list one phoneme in the overlap area"

    overlap_duration = info_taken_phonemes[desirable_phonemes-1].maxTime*16000 - start_p
    i=2
    while overlap_duration > window_length/2 and desirable_phonemes-i>=0:
        overlap_duration = info_taken_phonemes[desirable_phonemes-i].maxTime*16000 - start_p
        i+=1

    overlap_duration =  min(int(round(overlap_duration)),  int(round(window_length/2)))
    return overlap_duration
 
def sample_segment(audio, n_samples, ret_idx=False):
    """
    samples a random segment of `n_samples` from `audio`.
    if audio is shorter than `n_samples` then return unchanged.
    audio - tensor of shape [1, T]
    n_samples - int, this will be the new length of audio
    ret_idx - if True then the start and end indices will be returned
    """
    if audio.shape[1] > n_samples:
        diff = audio.shape[1] - n_samples
        start = random.randint(0, diff)
        end = start + n_samples
        new_audio = audio[:, start:end]
        if ret_idx:
            return new_audio, (start, end)
        return new_audio

    if ret_idx:
        return audio, (0, audio.shape[1] - 1)
    return audio

def build_phoneme_and_energy_representation(total_phoneme_len, phoneme_numer, cur_energy):

    ## represent by repeating the phoneme_number devided by the total num of the phonemes. 
    represent_sign = phoneme_numer / TOTAL_AVAILABLE_PHONEMES
    phonemes_representaion = torch.Tensor.repeat(torch.tensor(represent_sign),math.floor(total_phoneme_len))
    energy_representaion = torch.Tensor.repeat(torch.tensor(cur_energy),math.floor(total_phoneme_len))
    return phonemes_representaion, energy_representaion

def Build_excisting_phonemes_sec_approach(start_frame, end_frame ,phonemes_a, Energy_a, sr):
    list_taken_phonemes = []
    list_info_taken_phonemes = []
    conditioned_phonemes_signal = torch.empty((0)) 
    conditioned_energy_signal = torch.empty((0)) 
    for phoneme, cur_energy in zip(phonemes_a,Energy_a) :
        start_phoneme = round(phoneme.minTime * sr)
        end_phoneme = round(phoneme.maxTime * sr)
        phoneme_mark = phoneme.mark
        if start_phoneme >= end_frame : # --> start_frame < start_phoneme
            break
        elif end_phoneme <= start_frame : # --> end_frame>end_phoneme
            continue
        elif start_phoneme >= start_frame:
            if end_phoneme<= end_frame:
                total_phoneme_length = round(end_phoneme-start_phoneme)
                list_taken_phonemes.append([phoneme_to_index_dict[phoneme_mark],total_phoneme_length])
                list_info_taken_phonemes.append(phoneme)
                cur_phoneme_representaion, cur_energy_representation = build_phoneme_and_energy_representation(total_phoneme_length,
                                                phoneme_to_index_dict[phoneme_mark], cur_energy)
                conditioned_phonemes_signal = torch.concat((conditioned_phonemes_signal,cur_phoneme_representaion))
                conditioned_energy_signal = torch.concat((conditioned_energy_signal,cur_energy_representation))
                continue
            else: # --> end_phoneme > end_frame
                total_phoneme_length = round(end_frame-start_phoneme)
                list_taken_phonemes.append([phoneme_to_index_dict[phoneme_mark],total_phoneme_length])
                list_info_taken_phonemes.append(phoneme)
                cur_phoneme_representaion, cur_energy_representation = build_phoneme_and_energy_representation(total_phoneme_length,
                                                phoneme_to_index_dict[phoneme_mark], cur_energy)
                conditioned_phonemes_signal = torch.concat((conditioned_phonemes_signal,cur_phoneme_representaion))
                conditioned_energy_signal = torch.concat((conditioned_energy_signal,cur_energy_representation))
                continue
        elif start_phoneme <= start_frame:
            if end_phoneme >= end_frame:
                total_phoneme_length = round(end_frame-start_frame)
                list_taken_phonemes.append([phoneme_to_index_dict[phoneme_mark],total_phoneme_length])
                list_info_taken_phonemes.append(phoneme)
                cur_phoneme_representaion, cur_energy_representation = build_phoneme_and_energy_representation(total_phoneme_length,
                                                phoneme_to_index_dict[phoneme_mark], cur_energy)
                conditioned_phonemes_signal = torch.concat((conditioned_phonemes_signal,cur_phoneme_representaion))
                conditioned_energy_signal = torch.concat((conditioned_energy_signal,cur_energy_representation))
                continue
            else: # --> ende_frame > end_phonem
                total_phoneme_length = round(end_phoneme-start_frame)
                list_taken_phonemes.append([phoneme_to_index_dict[phoneme_mark],total_phoneme_length])
                list_info_taken_phonemes.append(phoneme)
                cur_phoneme_representaion, cur_energy_representation = build_phoneme_and_energy_representation(total_phoneme_length,
                                                phoneme_to_index_dict[phoneme_mark], cur_energy)
                conditioned_phonemes_signal = torch.concat((conditioned_phonemes_signal,cur_phoneme_representaion))
                conditioned_energy_signal = torch.concat((conditioned_energy_signal,cur_energy_representation))
                continue
        else:
            assert  False, "Roi you missed at least one case."

    tensor_taken_phonemes = torch.tensor(list_taken_phonemes)
    ## The int, round, came after the run training
    assert conditioned_phonemes_signal.shape[0] == int(round(end_frame-start_frame)), f"some how {conditioned_phonemes_signal.shape[0]} and {end_frame-start_frame} are different"
    assert conditioned_phonemes_signal.shape[0] == conditioned_energy_signal.shape[0], f"some how {conditioned_phonemes_signal.shape[0]} and {conditioned_energy_signal.shape[0]} are different"
    assert tensor_taken_phonemes.shape[0]>0 , "There are no phonemes in the phonemes tensor."
    return conditioned_phonemes_signal[None,:], list_info_taken_phonemes, conditioned_energy_signal[None,:]

class TextGridDataset(torch.utils.data.Dataset):
    def __init__(self, json_manifest_TextGrids):
        self.files = json.load(open(json_manifest_TextGrids, "r"))
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self,i):   
        path = self.files[i]
        tg = textgrid.TextGrid.fromFile(path)

        return tg

class Npy_EnergyDataset(torch.utils.data.Dataset):
    def __init__(self, json_manifest_npy_energy):
        self.files = json.load(open(json_manifest_npy_energy, "r"))
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self,i):   
        path = self.files[i]
        cur_energy = np.load(path)

        return cur_energy
    
class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, json_manifest, n_samples=None, min_duration=0, max_duration=float("inf")):
        if n_samples:
            assert n_samples <= min_duration, "`min_duration` must be greater than `n_samples`"
        self.n_samples = n_samples

        # load list of files
        logger.info(f"loading from: {json_manifest}")
        self.files = json.load(open(json_manifest, "r"))
        logger.info(f"files in manifest: {len(self.files)}")
        # filter files that are with Inappropriate duration
        self.files = list(filter(lambda x: min_duration <= x[1] <= max_duration, self.files))
        logger.info(f"files after duration filtering: {len(self.files)}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        path, length = self.files[i]
        audio, sr = torchaudio.load(path)

        if self.n_samples:
            audio = sample_segment(audio, self.n_samples)

        return audio

class PairedAudioDataset(torch.utils.data.Dataset):
    def __init__(self, json_wav, json_TextGrids, json_npy_Energy,  n_samples=None, min_duration=0, max_duration=float("inf")):

        if n_samples:
            assert n_samples <= min_duration, "`min_duration` must be greater than `n_samples`"
        self.n_samples = n_samples

        self.ds_a = AudioDataset(
            json_manifest=json_wav,
            n_samples=None, 
            min_duration=min_duration,
            max_duration=max_duration,
        )
       
        self.ds_b = AudioDataset(
            json_manifest=json_wav,
            n_samples=None, 
            min_duration=min_duration,
            max_duration=max_duration,
        )

        self.ds_textgrids_a = TextGridDataset(
            json_manifest_TextGrids=json_TextGrids,
        )

        self.ds_energy_a = Npy_EnergyDataset(
            json_manifest_npy_energy = json_npy_Energy,
        )

        assert len(self.ds_a) == len(self.ds_b), "datasets in `PairedAudioDataset` must be of equal length"

    def __len__(self):
        return len(self.ds_a)

    def __getitem__(self, i):
        # Using the same index for iterators.
        audio_a = self.ds_a[i]
        audio_b = self.ds_b[i]
        phonemes_a =  self.ds_textgrids_a[i][1]
        Energy_a = self.ds_energy_a[i]
        if self.n_samples:
            # sample identically for both waveforms (calling to PairedAudioDataset)
            # (using n_samples != None)
            audio_a, (start, end) = sample_segment(audio_a, self.n_samples, ret_idx=True)

        conditioned_phonemes_signal_a, list_info_cur_taken_phonemes, conditioned_energy_signal_a = \
            Build_excisting_phonemes_sec_approach(start, end ,phonemes_a,Energy_a,16000)

        if self.n_samples:
        ###The overlap is by num of samples and not by time.
            overlap_zone = int(get_overlap_duration(start, list_info_cur_taken_phonemes, 8000))
            audio_b = audio_b[:, start:end]
            audio_b[:,overlap_zone:]=0

        return audio_a, audio_b, conditioned_phonemes_signal_a, conditioned_energy_signal_a
    
if __name__ == "__main__":
    print("audio.py")
