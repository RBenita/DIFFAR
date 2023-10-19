import torch
import numpy as np
import json
import torchaudio
import textgrid
import os
import csv
phoneme_to_index_dict = { "" : 1, "AA0": 2, 'AA1': 3, 'AA2': 4, 'AE0': 5 ,'AE1': 6
 ,'AE2': 7 ,'AH0': 8 ,'AH1': 9 ,'AH2': 10 ,'AO0': 11 ,'AO1': 12 ,'AO2': 13 ,'AW0': 14 ,'AW1': 15
   ,'AW2': 16 ,'AY0': 17 ,'AY1': 18 ,'AY2': 19,'B': 20 ,'CH': 21, 'D' : 22, 'DH' :23,
   'EH0':24, 'EH1':25, 'EH2':26 ,'ER0':27, 'ER1':28,'ER2':29 ,'EY0':30 , 'EY1':31 ,'EY2':32 ,'F':33 ,
   'G':34 ,'HH':35 ,'IH0':36 ,'IH1':37 ,'IH2':38 ,'IY0':39, 'IY1':40 , 'IY2':41 , 'JH':42,
    'K':43,  'L':44,  'M':45,  'N':46 , 'NG':47,  'OW0':48, 'OW1':49, 'OW2':50,'OY0':51, 'OY1':52, 
    'OY2':53, 'P':54, 'R':55, 'S':56, 'SH':57, 'T':58, 'TH':59, 'UH0':60, 'UH1':61,
     'UH2':62, 'UW0':63, 'UW1':64, 'UW2':65, 'V':66, 'W':67, 'Y':68, 'Z':69, 'ZH':70, 'spn':71, 'sil':72
}

def get_phonemes_from_file(file_pass_,phonemes):
    Phonemes_list = ""
    Duration_list = ""
    for curr_phoneme_info in phonemes:
        Phonemes_list= Phonemes_list + "," + curr_phoneme_info.mark
        total_cur_phoneme_dur = round(curr_phoneme_info.maxTime - curr_phoneme_info.minTime,2)
        Duration_list = Duration_list + "," + str(total_cur_phoneme_dur)
    Duration_list = Duration_list[1:]
    Phonemes_list = Phonemes_list[1:]
    return Phonemes_list, Duration_list

def producing_csv_file(Text_grid_json_file, wav_files_path, csv_filename, csv_path, sr=16000):
    ## get a json file of textgrid files
    ## Output csv file whichincludes: wav_path_name, phonemes, duration.
    text_grid_files = json.load(open(Text_grid_json_file, "r"))
    total_data =[]
    total_file_names = []
    print(len(text_grid_files))

    # csv header
    fieldnames = ['wav_path_name','sr','phonemes', 'duration']
    for file_path in (text_grid_files):
        # print(file_path) 
        current_phonemes = textgrid.TextGrid.fromFile(file_path)[1]
        Phonemes_list, Duration_list = get_phonemes_from_file(file_path, current_phonemes)
        # row = {'phonemes': Phonemes_list, 'duration': Duration_list}
        # print(f"Phonemes_list: {Phonemes_list}, Duration_list: {Duration_list} ")
        # total_data.append(row)

        file_name = os.path.basename(file_path)
        file_name = os.path.splitext(file_name)[0]
        cur_wav_file_path= wav_files_path + file_name + "_16000.wav"
        total_data.append([cur_wav_file_path, sr, Phonemes_list, Duration_list])

    with open(f'{csv_path}/{csv_filename}.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        # write the header
        header = ["wav_path_name", "sr", "phonemes", "duration"]
        writer.writerow(header)
        # write multiple rows
        writer.writerows(total_data)
  
class Accuracy:
    def __init__(self):
        self.y, self.yhat = [], []

    def update(self, yhat, y):
        self.yhat.append(yhat)
        self.y.append(y)

    def acc(self, tol):
        yhat = torch.cat(self.yhat)
        y = torch.cat(self.y)
        acc = torch.abs(yhat - y) <= tol
        acc = acc.float().mean().item()
        return acc

if __name__ == '__main__':
    print("energy_predictor_utils")

    
    







