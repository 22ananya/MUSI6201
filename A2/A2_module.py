# Module with functions for A2

# load dependencies
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
# import helper function module from A1
from A1_helper_module import *
import os
import scipy.io.wavfile as wav

# Inputs:
fs = 44.1e3 # sampling rate
block_size = 1024 # block size
hop_size = 512 # hop size

# load audio and corresponding text data files from directory
# set absolute path to directory containing all files
musicpath = r'/Users/ananyabhardwaj/Downloads/music_speech data/music_wav' # update as required for your system
speechpath = r'/Users/ananyabhardwaj/Downloads/music_speech data/speech_wav' # update as required for your system

# create spectral centroid function - takes blocked audio and sampling rate as input
def extract_spectral_centroid(xb, fs):
    # calculate fft block by block, keeping only the positive frequencies and normalizing by the number of samples
    # create a frequency vector
    nblocks, b_size = np.shape(xb)
    f = np.arange(0,fs/2 + fs/b_size,fs/b_size)
    flen = len(f)
    hw = np.hanning(b_size) # hanning window

    X_fft = np.zeros((nblocks, flen)) # FFT vector
    S_cent = np.zeros(nblocks) # spectral centroid
    for i in range(nblocks):
        X_fft[i,:] = (np.abs(np.fft.fft(hw*xb[i,:]))[0:flen])/(0.5*b_size)
        S_cent[i] = (np.dot(f,X_fft[i,:]))/np.sum(X_fft[i,:]) # spectral centroid
    return S_cent

# calculate RMS energy in dB
def extract_rms(xb):
    nblocks, b_size = np.shape(xb) # get size of input vector
    rms_dB = np.zeros(nblocks) # spectral centroid
    for i in range(nblocks):
        rms_dB[i] = np.maximum(20*np.log10(np.sqrt(np.sum(np.square(xb[i,:])/np.size(xb[i,:])))), -100)
    return rms_dB
        


# Create a function for extracting zero_crossings
def extract_zerocrossingrate(xb):
    # loop over each block and find zero crossing rate by finding negative difference values
    nblocks, b_size = np.shape(xb) # get size of input vector
    zcr = np.zeros(nblocks) # spectral centroid
    for i in range(nblocks):
        zcr[i] = (1/(2*b_size))*np.sum( np.abs(np.diff(np.sign(xb[i,:]))))
    return zcr

# create a function to calculate spectral crest
def extract_spectral_crest(xb, fs):
    # calculate fft block by block, keeping only the positive frequencies and normalizing by the number of samples
    # create a frequency vector
    nblocks, b_size = np.shape(xb)
    f = np.arange(0,fs/2 + fs/b_size,fs/b_size)
    flen = len(f)
    hw = np.hanning(b_size) # hanning window

    X_fft = np.zeros((nblocks, flen)) # FFT vector
    S_crest = np.zeros(nblocks) # spectral centroid
    for i in range(nblocks):
        X_fft[i,:] = (np.abs(np.fft.fft(hw*xb[i,:]))[0:flen])/(0.5*b_size)
        S_crest[i] = (np.max(X_fft[i,:]))/np.sum(X_fft[i,:]) # spectral crest
    return S_crest

# create a function to calculate spectral flux
def extract_spectral_flux(xb, fs):
    # calculate fft block by block, keeping only the positive frequencies and normalizing by the number of samples
    # create a frequency vector
    nblocks, b_size = np.shape(xb)
    f = np.arange(0,fs/2 + fs/b_size,fs/b_size)
    flen = len(f)
    hw = np.hanning(b_size) # hanning window

    X_fft = np.zeros((nblocks, flen)) # FFT vector
    S_flux = np.zeros(nblocks) # spectral centroid
    for i in range(nblocks-1):
        X_fft[i,:] = (np.abs(np.fft.fft(hw*xb[i,:]))[0:flen])/(0.5*b_size)
        X_fft[i+1,:] = (np.abs(np.fft.fft(hw*xb[i+1,:]))[0:flen])/(0.5*b_size)
        S_flux[i] = np.sqrt(np.sum(np.square(X_fft[i+1,:] - X_fft[i,:])))/ (b_size+1) # spectral flux
    return S_flux








                                                  ## Plotting and Testing

# create a sample signal
t = np.arange(0,1,1/fs) # time vector
x = np.sin(2*np.pi*441*t) #+ np.sin(2*np.pi*880*t) + np.sin(2*np.pi*1320*t)  # signal vector
# add some filtered white noise to signal vector
noise = np.random.normal(0,1,len(x))
b, a = signal.butter(2, 15e3, 'low', fs = fs, analog=False) # to prevent aliasing, filter noise before adding to signal
noise = signal.filtfilt(b, a, noise)
#x = x + 0.1*noise


def extract_features(x, block_size, hop_size, fs):
    # First block an input signal
    xb, timeInSec = block_audio(x,block_size, hop_size, fs)
    NumOfBlocks = len(timeInSec)

    # create an empty feature array
    feature_array = np.zeros((NumOfBlocks, 5))

    # calculate spectral centroid
    S_cent = extract_spectral_centroid(xb, fs)
    feature_array[:,0] = S_cent

    # calculate rms_dB
    rms_dB = extract_rms(xb)
    feature_array[:,1] = rms_dB

    # calculate spectral crest
    S_crest = extract_spectral_crest(xb, fs)
    feature_array[:,2] = S_crest

    # calculate zero crossings
    zcr = extract_zerocrossingrate(xb)
    feature_array[:,3] = zcr

    # calculate spectral flux
    S_flux = extract_spectral_flux(xb, fs)
    feature_array[:,4] = S_flux

    return feature_array, timeInSec

def aggregate_feature_per_file(features):
    # aggregate features per file by taking the mean and standard deviation of each feature
    agg_features = np.zeros(10) # initialize empty array
    # loop over features to calculate mean and standard deviation
    c = 0
    for i in range(np.shape(features)[1]):
        agg_features[c],agg_features[c+1]  =   np.mean(features[:,i]), np.std(features[:,i])
        c = c+2
    return agg_features # remove first row of zeros


# create a function that loops over all the files in a given folder path, extracts and saves all the feature values in one vector
def get_feature_data(path, block_size, hop_size):
    audio_files = [] # initialize list of audio files
    
    for file in os.listdir(path): # iterate over all files in the directory
        if file.endswith('.wav'): # if the file is an audio file
            audio_files.append(os.path.join(path, file)) # add the file to the list of audio files

    N = len(audio_files) # length of audio files in given folder
    feature_data = np.zeros((10, N)) # create an array of zeros of the size of 10 (number of feature values) by N (number of files in the folder)
    
    # loop over all audio files and process them to extract features that are aggregated in the feature_data array
    for i in range(N):
        print('Processing file ' + str(i+1) + ' of ' + str(len(audio_files)) + '...') # print progress
        fs, x = wav.read(audio_files[i]) # load audio file

        #  extract features from one file
        feature_array, timeInSec = extract_features(x, block_size, hop_size, fs)
        # get means and std of features and save to aggrgate feature vector
        feature_data[:,i] = aggregate_feature_per_file(feature_array)

    return feature_data

                                        ## Part B

# create a z-score normalization function - normalizes each feature to zero mean and 1 std
def normalize_zscore(features):
    features_zsc = np.zeros(shape=np.shape(features))
    for i in range(np.shape(features)[1]):
        features_zsc[:,i] = (features[:,i] - np.mean(features[:,i]))/np.std(features[:,i])
    
    return features_zsc


                                        ## Part C

def visualize_features(path_to_musicspeech):
    block_size = 1024 # block size
    hop_size = 256 # hop size
    musicpath = path_to_musicspeech + 'music_wav' # update as required for your system
    speechpath = path_to_musicspeech + 'speech_wav' # update as required for your system

    feature_data_music = get_feature_data(musicpath, block_size, hop_size)
    feature_data_speech = get_feature_data(speechpath, block_size, hop_size)

    N_music = np.shape(feature_data_music)[1]
    N_speech = np.shape(feature_data_speech)[1]

    # normalize features across both music and speech
    # combine music and speech feature data
    feature_data = np.concatenate((feature_data_music, feature_data_speech), axis=1)
    # normalize features
    feature_data_zsc = normalize_zscore(feature_data)
    # separate music and speech features
    feature_data_music_zsc = feature_data_zsc[:,0:N_music]
    feature_data_speech_zsc = feature_data_zsc[:,N_music:]

    ## visualize and compare features across music and speech using scatter plots
    # spectral centroid  vs spectral crest means
    plt.figure()
    plt.scatter(feature_data_music_zsc[0,:], feature_data_music_zsc[4,:], c='r', marker='o', label='music')
    plt.scatter(feature_data_speech_zsc[0,:], feature_data_speech_zsc[4,:], c='b', marker='x', label='speech')
    plt.xlabel('Spectral Centroid')
    plt.ylabel('Spectral Crest')
    plt.title('Spectral Centroid vs Spectral Crest')
    plt.legend()
    plt.show(block=False)

    # spectral flux vs zero crossings means
    plt.figure()
    plt.scatter(feature_data_music_zsc[8,:], feature_data_music_zsc[6,:], c='r', marker='o', label='music')
    plt.scatter(feature_data_speech_zsc[8,:], feature_data_speech_zsc[6,:], c='b', marker='x', label='speech')
    plt.xlabel('Spectral Flux')
    plt.ylabel('Zero Crossings')
    plt.title('Spectral Flux vs Zero Crossings')
    plt.legend()
    plt.show(block=False)

    # RMS mean vs RMS std
    plt.figure()
    plt.scatter(feature_data_music_zsc[2,:], feature_data_music_zsc[3,:], c='r', marker='o', label='music')
    plt.scatter(feature_data_speech_zsc[3,:], feature_data_speech_zsc[3,:], c='b', marker='x', label='speech')
    plt.xlabel('RMS mean')
    plt.ylabel('RMS std')
    plt.title('RMS mean vs RMS std')
    plt.legend()
    plt.show(block=False)

    # Zero crossing std vs spectral crest std
    plt.figure()
    plt.scatter(feature_data_music_zsc[7,:], feature_data_music_zsc[5,:], c='r', marker='o', label='music')
    plt.scatter(feature_data_speech_zsc[7,:], feature_data_speech_zsc[5,:], c='b', marker='x', label='speech')
    plt.xlabel('Zero Crossing std')
    plt.ylabel('Spectral Crest std')
    plt.title('Zero Crossing std vs Spectral Crest std')
    plt.legend()
    plt.show(block=False)

    # spectral centroid std vs spectral flux std
    plt.figure()
    plt.scatter(feature_data_music_zsc[1,:], feature_data_music_zsc[9,:], c='r', marker='o', label='music')
    plt.scatter(feature_data_speech_zsc[1,:], feature_data_speech_zsc[9,:], c='b', marker='x', label='speech')
    plt.xlabel('Spectral Centroid std')
    plt.ylabel('Spectral Flux std')
    plt.title('Spectral Centroid std vs Spectral Flux std')
    plt.legend()
    plt.show()

    return




# plot input signal
plt.figure()
plt.plot(t,x)
plt.xlabel('Time (sec)')
plt.ylabel('Magnitude')
plt.title('Input Signal')
plt.show(block=False)

plt.figure()
plt.plot(t,noise)
plt.xlabel('Time (sec)')
plt.ylabel('Magnitude')
plt.title('Noise')
plt.show(block=False)

# plot sample spectrum
# plt.figure()
# plt.plot(f, X_fft[4,:])
# plt.xlabel('Frequency (Hz)')
# plt.ylabel('Magnitude')
# plt.title('Sample Spectrum')
# plt.show(block=False)

# plot spectrogram
# plt.figure()
# plt.pcolormesh(X_fft.T, shading='flat')
# plt.colorbar()
# plt.xticks(np.arange(0,NumOfBlocks,10),np.round(timeInSec[0:NumOfBlocks:10],1))
# plt.yticks(np.arange(0,flen,100),np.round(f[0:flen:100]))
# plt.xlabel('Time (sec)')
# plt.ylabel('Frequency (Hz)')
# plt.title('Spectrogram')
# plt.show(block=False)


path_musicspeech = r'/Users/ananyabhardwaj/Downloads/music_speech data/' # update as required for your system                                      
visualize_features(path_musicspeech)

## Uncomment below to visualize if needed

# # plot spectral centroid vector
# plt.figure()
# plt.plot(timeInSec, features[:,0])
# plt.xlabel('Time (sec)')
# plt.ylabel('Magnitude')
# plt.title('Spectral Centroid')
# plt.show(block=False)

# # plot RMS Energy
# plt.figure()
# plt.plot(timeInSec, features[:,1])
# plt.xlabel('Time (sec)')
# plt.ylabel('Magnitude')
# plt.title('RMS [dB]')
# plt.show(block=False)

# # plot zerocrossings
# plt.figure()
# plt.plot(timeInSec, features[:,2])
# plt.xlabel('Time (sec)')
# plt.ylabel('Magnitude')
# plt.title('Zero Crossings')
# plt.show(block=False)

# # plot spectral crest
# plt.figure()
# plt.plot(timeInSec, features[:,3])
# plt.xlabel('Time (sec)')
# plt.ylabel('Magnitude')
# plt.title('Spectral Crest')
# plt.show(block=False)

# # plot spectral flux
# plt.figure()
# plt.plot(timeInSec, features[:,4])
# plt.xlabel('Time (sec)')
# plt.ylabel('Magnitude')
# plt.title('Spectral Flux')
# plt.show()