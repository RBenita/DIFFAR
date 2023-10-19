import os
import textgrid
import torch
from duration_predictors_files.playground_duration import grapheme_to_phoneme_func, making_textgrid_file_start_with_None
from duration_predictors_files.duration_predictor import load_ckpt, phoneme_to_index_dict

def get_phonemes_list_from_file(Text_grid_path_file):
    current_phonemes = textgrid.TextGrid.fromFile(Text_grid_path_file)[1]
    Phonemes_list = []
    for curr_phoneme_info in current_phonemes:
        Phonemes_list.append(curr_phoneme_info.mark)
    print(Phonemes_list)
    return Phonemes_list

def producing_filename_text_phoneme_list(origin_text_path, original_textgrids_path=None):
    """
    move through all text files in a givem path.
    make a list which consist of ["file name","text in the file"]
    """
    list_info = []
    path_of_the_directory = origin_text_path
    ext = ('.txt')
    for files in os.listdir(path_of_the_directory):
        if files.endswith(ext):
            # print(files) 
            file_name = os.path.basename(files)
            file_name = os.path.splitext(file_name)[0]
            with open(f"{origin_text_path}/{files}", 'r') as file:
                data = file.read().rstrip()

            ### using phonemes from G2P MODEL
            if original_textgrids_path == None:
                out_phonemes=grapheme_to_phoneme_func([data])

            ### using original phonemes from original Textgrid
            else:
                textgrid_file_path = f"{original_textgrids_path}/{file_name}.TextGrid"
                out_phonemes =  get_phonemes_list_from_file(textgrid_file_path)
        
            list_info.append([file_name,data,out_phonemes])
    return list_info

def test_on_given_lists(cur_list, target_predicted_textgrid_path,model_path=None,Is_G2P=False):
    device = "cuda:0"
    i=0
    cur_model = load_ckpt(model_path)
    cur_model = cur_model.to(device)
   
    for curr_sample in cur_list:
        cur_file_name, curr_text, cur_phoneme_original = curr_sample[0], curr_sample[1], curr_sample[2]
        cur_phoneme = [phoneme_to_index_dict[x] for x in cur_phoneme_original]
        cur_phoneme = torch.LongTensor(cur_phoneme).to(device)[None,:]
        duration_hat = cur_model(cur_phoneme)
        duration_hat_scaled = torch.exp(duration_hat) - 1
        duration_hat_by_sec = duration_hat_scaled / 100
        target_path = f"{target_predicted_textgrid_path}/predicted_{cur_file_name}.TextGrid"
        making_textgrid_file_start_with_None(duration_hat_by_sec,cur_phoneme_original,target_path, Is_G2P)
        i+=1

if __name__ == '__main__':
    ### using G2P MODEL
    path_target_predicted_textgrid = "/home/mlspeech/rbenita/PycharmProjects/git_models/DiffAR_200/duration_predictors_files/textgrid_files"
    origin_text_path_ = "/home/mlspeech/rbenita/PycharmProjects/git_models/DiffAR_200/duration_predictors_files/text_files"
    file_names_and_text_list = producing_filename_text_phoneme_list(origin_text_path_, None)
    Is_G2P = True
   
    ### using original phonemes from original Textgrid
    # path_target_predicted_textgrid = "/home/mlspeech/rbenita/PycharmProjects/Duration_predictor/results_delete_after_finishes_original_ph_pred_dur"
    # origin_text_path = "/home/data/rbenita/data_for_test_results/original_text_files"
    # origin_textgrid_path = "/home/data/rbenita/data_for_test_results/original_textgrids"
    # file_names_and_text_list = producing_filename_text_phoneme_list(origin_text_path, origin_textgrid_path)
    # Is_G2P = False
    model_path = "/home/mlspeech/rbenita/PycharmProjects/git_models/DiffAR_200/duration_predictors_files/saved_models/predictor_model_sec_dividing_kernel_5.ckpt"
    test_on_given_lists(file_names_and_text_list,path_target_predicted_textgrid,model_path,Is_G2P)

    
