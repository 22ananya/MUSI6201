This is a readme for this project and how to run the code for evaluation

# Accessing Data
All of the code is in the form of Colab Notebooks, and all of the data (>100 GB) is hosted on my Google Drive. The folder is called "Cadenza Challenge"
I'll have shared a link to the Google Drive folder with you, you'll need to accept that link. Then, you'll need to go to "Shared With Me" in your google drive, right-click on the shared folder, select 
Organize->Create Shortcut->     Place that shortcut somewhere in your Drive, ideally, in the MyDrive location.

Then, all you have to do is run the Colab files as is (and Mount *your* GDrive - the code already does that) and you'll be able to run everything. And you'll have access to all the data.


# Prerequisites
The Colab notebooks will import anything required, so you can just hit run all cells and verify it all.

However, one primary / critical import is Pyclarity : https://github.com/claritychallenge/clarity/tree/main
I install it in every notebook as !pip install pyclarity==0.4.0 

This version was suggested by the challenge creators, so leave that unchanged.


# Files to run

Only look at files that start with FINAL. Everything else is likely just older and duplicate. 

Although all of the files are relevant, there are a few duplicates, so run the following and all of the project should make sense. The suggested order should make sense, but you don't have to follow it!

File 1
1. FINAL_Process_Tracks.ipynb - This script loads the at_mic_music (hrtf applied already), demixes it, reweights and downmixes the stems, applies hearing aid processing and saves audio to google drive as FLAC files. I load these FLAC files for analysis later on.

2. FINAL_Process_Reference_Stems.ipynb - This file does the same as 1, except it imports the reference stems, so doesn't do any demixing. The output of this script are the files that are used for evaluation with HAAQI and SI-SDR

3. FINAL_Run_HAAQI_Eval - This script will load the saved mixtures and reference mixtures, run HAAQI and save results

4. FINAL_Run_SI_SDR_Eval.ipynb - Same as 3, except here it does SI_SDR

5. FINAL_Visualization_and_Listening.ipynb - This script just loads and plots spectrograms and audio at each stage of processing - a way to evaluate performance subjectively

6. FINAL_Demixed_Stems_Listening.ipynb - This loads input mixtures, demixes with demucs or openunmix, and plays them back for inference

Optional

7. testing.ipynb - This is the script that was used to generate the at_mic_music for training

8. start.ipynb - This was my initial attempt at creating my own source separation model, I abandoned it fairly early

9. Retrain_Open_Unmix_Pytorch.ipynb - Fine tuning OpenUnmix took too long, so we didn't see it through


* All the files above work on the validation set, if you change the root to "/content/drive/MyDrive/Cadenza_Challenge/cad_icassp_2024" instead of "/content/drive/MyDrive/Cadenza_Challenge/cad_icassp_2024/Validation", then you can run everything on the training set, but full tracks are slow to process. Some folders are also named differently if you choose to do that, you can just run some of the older colab scripts that are for the train set. I've not updated them since there fundamentally is no advantage to it, other than slower inference. 

