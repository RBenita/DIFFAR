import os
import textgrid
import torch
import numpy as np
from energy_predictor_files.energy_predictor import load_ckpt, phoneme_to_index_dict
import torch.nn.functional as F

def get_phonemes_list_from_file(Text_grid_path_file):
    current_phonemes = textgrid.TextGrid.fromFile(Text_grid_path_file)[1]
    Phonemes_list = []
    for curr_phoneme_info in current_phonemes:
        Phonemes_list.append(curr_phoneme_info.mark)
    print(Phonemes_list)
    return Phonemes_list

def producing_phoneme_list(original_textgrids_path):
    """
    move through all text files in a givem path.
    make a list which consist of ["file name","phonemes in the file""]
    """
    list_info = []
    import os
    path_of_the_directory = original_textgrids_path
    ext = ('.TextGrid')
    for files in os.listdir(path_of_the_directory):
        if files.endswith(ext):
            file_name = os.path.basename(files)
            file_name = os.path.splitext(file_name)[0]

            ### using a given phonemes from Textgrid file
            textgrid_file_path = f"{original_textgrids_path}/{file_name}.TextGrid"
            out_phonemes =  get_phonemes_list_from_file(textgrid_file_path)
            list_info.append([file_name,out_phonemes])
    return list_info

def test_on_given_lists_energy(cur_list, target_predicted_textgrid_path, model_path=None):
    device = "cuda:0"
    i=0
    cur_model = load_ckpt(model_path)
    cur_model = cur_model.to(device)
    for curr_sample in cur_list:
        cur_file_name, cur_phoneme_original = curr_sample[0], curr_sample[1]
        cur_phoneme = [phoneme_to_index_dict[x] for x in cur_phoneme_original]
        cur_phoneme = torch.LongTensor(cur_phoneme).to(device)[None,:]
        energy_hat = cur_model(cur_phoneme)
        energy_hat_scaled = torch.exp(energy_hat) - 1
        energy_hat_by_original_scale = energy_hat_scaled / 100
        print(f"file name: {cur_file_name}")

        ## saving the predicted energy ##
        path_predicted = target_predicted_textgrid_path + "/predicted_Energy_" + cur_file_name + ".npy"
        enery_numpy = energy_hat_by_original_scale.cpu().detach().numpy()[0]
        np.save(path_predicted, enery_numpy)
        i+=1
    
if __name__ == '__main__':

    ### using phonemse from G2P MODEL
    path_target_predicted_energy = "/home/mlspeech/rbenita/PycharmProjects/git_models/DiffAR_200/energy_predictor_files/energy_files"
    G2P_textgrid_path = "/home/mlspeech/rbenita/PycharmProjects/git_models/DiffAR_200/duration_predictors_files/textgrid_files"
    file_names_and_phoneme_list = producing_phoneme_list(G2P_textgrid_path)

    ### using original phonemes from original Textgrid
    # path_target_predicted_energy = "/home/mlspeech/rbenita/PycharmProjects/energy_predictor_by_phonemes/pred_ener_or_[hon_delet_after_fini"
    # origin_textgrid_path = "/home/data/rbenita/data_for_test_results/65_attempts/original_textgrids"
    # file_names_and_phoneme_list = producing_phoneme_list(origin_textgrid_path)

    # print(file_names_and_text_list)
    model_path = "/home/mlspeech/rbenita/PycharmProjects/git_models/DiffAR_200/energy_predictor_files/saved_models/energy_predictor_model_2_layers_7_5_k.ckpt"
    test_on_given_lists_energy(file_names_and_phoneme_list,path_target_predicted_energy, model_path)
    print("hello")
    
