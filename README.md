# DiffAR

Abstract: Diffusion models have recently been shown to be relevant for high-quality speech generation. Most work has been focused on generating spectrograms, and as such, they further require a subsequent model to convert the spectrogram to a waveform (i.e., a vocoder). This work proposes a diffusion probabilistic end-to-end model for generating a raw speech waveform. The proposed model is autoregressive, generating overlapping frames sequentially, where each frame is conditioned on a portion of the previously generated one. Hence, our model can effectively synthesize an unlimited speech duration while preserving high-fidelity synthesis and temporal coherence. We implemented the proposed model for unconditional and conditional speech generation, where the latter can be driven by an input sequence of phonemes, amplitudes, and pitch values. Working on the waveform directly has some empirical advantages. Specifically, it allows the creation of local acoustic behaviors, like vocal fry, which makes the overall waveform sounds more natural. Furthermore, the proposed diffusion model is stochastic and not deterministic; therefore, each inference generates a slightly different waveform variation, enabling abundance of valid realizations. Experiments show that the proposed model generates speech with superior quality compared with other state-of-the-art neural speech generation systems.


### This repository is currently under construction. Currently, you can access synthesis examples in the "Examples" folder, and detailed code will be uploaded in the near future. ###

### An HTML file summarizing representative examples is available here: ###
[Open html](github_IO/index.html)

## TODO
- [] Inference Procedure for DiffAR 200
- [] Training procedure for DiffAR 200


## DataSets ##
Currently, The supported dataset is:

LJSpeech: [GitHub Pages](https://keithito.com/LJ-Speech-Dataset/) a single-speaker English dataset consists of 13100 short audio clips of a female speaker reading passages from 7 non-fiction books, approximately 24 hours in total.

## Preprocessing ##
Before training your model, make sure to have the following .json files:
1. _wav.json files:
```
- train_wav.json
- val_wav.json
```
An example can be found [here](https://github.com/RBenita/DIFFAR/blob/main/Demo_json_files/Demo_wav.json)


You can generate these files by running:
`python flder2json.py <wavs_directory> | <json_directory>`

2. _TextGride.json files:
```
- train_TextGrid.json
- val_TextGrid.json
```
An example can be found [here](https://github.com/RBenita/DIFFAR/blob/main/Demo_json_files/Demo_textgrid.json)


You can generate these files by running:
`python flder2json_txt.py <Textgrid_directory> | <json_directory>`

3. _Energy.json files:
```
- train_Energy.json
- val_Energy.json
```
An example can be found [here](https://github.com/RBenita/DIFFAR/blob/main/Demo_json_files/Demo_npy_energy.json)


You can generate these files by:
   * Make a folder with energy .npy  files using the function:  from_wav_file_to_npy_energy_file
   * Run `python flder2json_npy.py <Energy_directory> | <json_directory>`

## Training ##
Our implementation is Using hydra.

* Make sure you have updated the [conf.yaml](https://github.com/RBenita/DIFFAR/blob/main/conf/conf.yaml) file correctly. Mainly pay attention to the pathes fields:
  ```
- train_ds
- valid_ds
```



## Infernece ##
To synthesize your custom .wav files: 
1. Locate the .txt files under a folder named 'text_files' as follows:
```
|-- current_directory
|   |-- text_files
|   |   |-- file1.txt
|   |   |-- file2.txt
|   |   |-- file3.txt
```
   
2. run `python inference.py --main_directory <current_directory>`

A successful run should yield the following folder structure:

```
|-- current_directory
|   |-- text_files
|   |   |-- file1.txt
|   |   |-- file2.txt
|   |   |-- file3.txt
|   |-- predicted_energy_files
|   |   |-- file1.npy
|   |   |-- file2.npy
|   |   |-- file3.npy
|   |-- predicted_TextGrid_files
|   |   |-- file1.TextGrid
|   |   |-- file2.TextGrid
|   |   |-- file3.TextGrid
|   |-- generated_wavs
|   |   |-- file1.wav
|   |   |-- file2.wav
|   |   |-- file3.wav

```




