## First module for Assignment A4 - Creates functions for Tuning Frequency Estimation

# import dependencies
import numpy as np
import scipy as sp
from scipy.io import wavfile as wav
import matplotlib.pyplot as plt
import os
import math

# add path to A1 and A2 modules
import sys
path = os.getcwd()
print(path)
# sys.path.append(path + '/A1 Pitch Tracking')
# sys.path.append(path + '/A2')
# sys.path.append(path + '/A3')
# from A1_helper_module import *
# from a2solution import *
# from a3module import *
# from convert_freq2midi import *

# define functions from provided solutions for prior assignments
def block_audio(x,blockSize,hopSize,fs):
    # allocate memory
    numBlocks = math.ceil(x.size / hopSize)
    xb = np.zeros([numBlocks, blockSize])
    # compute time stamps
    t = (np.arange(0, numBlocks) * hopSize) / fs
    x = np.concatenate((x, np.zeros(blockSize)),axis=0)
    for n in range(0, numBlocks):
        i_start = n * hopSize
        i_stop = np.min([x.size - 1, i_start + blockSize - 1])
        xb[n][np.arange(0,blockSize)] = x[np.arange(i_start, i_stop + 1)]
    return (xb,t)

def compute_hann(iWindowLength):
    return 0.5 - (0.5 * np.cos(2 * np.pi / iWindowLength * np.arange(iWindowLength)))

def compute_spectrogram(xb,fs):
    numBlocks = xb.shape[0]
    afWindow = compute_hann(xb.shape[1])
    X = np.zeros([math.ceil(xb.shape[1]/2+1), numBlocks])
    for n in range(0, numBlocks):
    # apply window
        tmp = abs(sp.fft.fft(xb[n,:] * afWindow))*2/xb.shape[1]
        # compute magnitude spectrum
        X[:,n] = tmp[range(math.ceil(tmp.size/2+1))]
        X[[0,math.ceil(tmp.size/2)],n]= X[[0,math.ceil(tmp.size/2)],n]/np.sqrt(2)
    #let's be pedantic about normalization
    f = np.arange(0, X.shape[0])*fs/(xb.shape[1])
    return (X,f)

def convert_freq2midi(fInHz, fA4InHz = 440):
    def convert_freq2midi_scalar(f, fA4InHz):
        if f.any() <= 0:
            return 0
        else:
            return (69 + 12 * np.log2(f/fA4InHz))
    fInHz = np.asarray(fInHz)
    if fInHz.ndim == 0:
        return convert_freq2midi_scalar(fInHz,fA4InHz)
    midi = np.zeros(fInHz.shape)
    for k,f in enumerate(fInHz):
        midi[k] = convert_freq2midi_scalar(f,fA4InHz)
    return (midi)


def get_spectral_peaks(X):
    """
    Returns the 20 largest spectral peaks of a signal - for a given column X of a spectrogram
    """
    # get the 20 largest peaks
    peakinds, _ = sp.signal.find_peaks(X, distance=1, height=None) # adjust distance and height for avoiding neighboring frequencies - depends on spectral resolution, to adjust
    # sort the peaks by magnitude
    peakinds = peakinds[np.argsort(X[peakinds])][::-1]
    # return the 20 largest peaks
    peaks = peakinds[:20]
    return peaks  

def estimate_tuning_freq(x, blockSize, hopSize, fs):
    # block signal
    xb, timeInSec = block_audio(x,block_size, hop_size, fs) # imported from A1 helper module
    X, fInHz = compute_spectrogram(xb, fs) # imported from A3 module 

    # get the 20 largest peaks for each block
    peakfreqs = np.apply_along_axis(get_spectral_peaks, 0, X)
    # get the frequencies of the peaks
    freqs = fInHz[peakfreqs]
    # convert the frequencies to midi pitch
    midiPitches = convert_freq2midi(freqs)
    # find the deviation from the nearest integer
    deviation = midiPitches - np.round(midiPitches)
    # convert the deviation to cents
    deviation = deviation * 100
    # create a histogram of the deviation and find the bin with the highest count
    hist, bin_edges = np.histogram(deviation, bins=100)
    # find the bin with the highest count
    maxbin = np.argmax(hist)
    # plot the histogram
    plt.figure()
    plt.plot(bin_edges[1:], hist)
    plt.xlabel('Deviation (cents)')
    plt.ylabel('Count')
    plt.title('Histogram of Deviation')
    plt.show()
    # return the bin with the highest count
    tuningFreq = bin_edges[maxbin] 
    # convert the tuning frequency to Hz
    tuningFreq = 440 * 2**(tuningFreq/1200)
    # return the tuning frequency
    return tuningFreq

if __name__ == '__main__':
    # execute pitch tracker on trainData dataset
    audiopath = r'/Users/ananyabhardwaj/Downloads/MUSI_6201/music_speech data/music_wav/classical2.wav' # update as required for your system

    # Test the function
    # load the audio file
    fs, x = wav.read(audiopath)
    #t = np.arange(0,10, 1/fs)
    #x = np.sin(2*np.pi*440*t)
    # set the block and hop sizes
    block_size = 2048
    hop_size = 1024
    # estimate the tuning frequency
    tuningFreq = estimate_tuning_freq(x, block_size, hop_size, fs)
    # print the tuning frequency
    print('Estimated tuning frequency: ', tuningFreq, 'Hz')
