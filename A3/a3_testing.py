## Main module for Assignment A3 - Includes all of the reuqisite functions

# import dependencies
import numpy as np
import scipy as sp
from scipy.io import wavfile as wav
import matplotlib.pyplot as plt
import os

# add path to A1 and A2 modules
import sys
path = os.getcwd()
sys.path.append(path + '/A1 Pitch Tracking')
sys.path.append(path + '/A2')
from A1_helper_module import *
from a2solution import *


def create_spectrogram(xb, fs):
    # compute the spectrogram of a given block of audio data xb sampled at a rate fs
    # xb: block of audio data
    # fs: sampling rate
    # returns: 2D array of spectrogram data
    # create a hann window of the same length as the block of audio data
    #xb = np.transpose(xb)
    NumOfBlocks, blockSize = np.shape(xb)
    hann_window = np.hanning(np.shape(xb)[1])
    hann_window = np.resize(hann_window, (np.shape(xb)))
    # multiply the window by the block of audio data
    xb = xb * hann_window
    # compute the fft of the block of audio data 
    fft = np.fft.fft(xb)
    # compute the magnitude of the fft
    magnitude = np.abs(fft)*(2/blockSize)
    # create a frequency vector
    fInHz = np.arange(0, fs/2+1, fs/blockSize)
    # compute the spectrogram from the fft, rejecting the second half of the fft
    #magnitude = np.transpose(magnitude)
    X = magnitude[:,0:blockSize//2+1]

    Y = np.transpose(X)

    plt.figure()
    plt.imshow(X)
    plt.show(block=False)

    plt.figure()
    plt.imshow(Y)
    plt.show(block=False)

    plt.figure()
    plt.plot(Y[:,5])
    plt.show()

    return Y, fInHz

def track_pitch_fftmax(x, blockSize, hopSize, fs):
    # block input audio vector x
    xb, timeInSec = block_audio(x, blockSize, hopSize, fs)
    # calculate magnitude spectrogram
    spect, freq = create_spectrogram(xb, fs)
    # find blockwise peak of spectrogram
    maxIndex = np.argmax(spect, axis=0)
    # return corresponding frequency vector
    return freq[maxIndex]


fs = 44.1e3
t1 = np.arange(0,1,1/fs)
t2 = np.arange(1,2,1/fs)
t = np.append(t1, t2)
f1 = 441
f2 = 882
x = np.append(np.sin(2*np.pi*f1*t1), np.sin(2*np.pi*f2*t2))
# plot test signal
plt.plot(t,x)
plt.xlabel('Time [s]')
plt.ylabel('Amplitude (raw)')
plt.title('Test Signal')
plt.show(block=False)

blockSize = 1024
hopSize = 512
#f0_fft = track_pitch_fftmax(x, blockSize, hopSize, fs)

xb, timeInSec = block_audio(x, blockSize, hopSize, fs)
# calculate magnitude spectrogram
# plt.figure()
# plt.plot(xb[0,:])
# plt.show(block=False)

# # apply hann window and plot
# hann_window = np.hanning(np.shape(xb)[1])
# hann_window = np.resize(hann_window, (np.shape(xb)))
# # multiply the window by the block of audio data
# xb = xb * hann_window

# plt.figure()
# plt.plot(xb[0,:])
# plt.show(block = False)

# plt.figure()
# plt.plot(xb[88,:])
# plt.show()






spect, freq = create_spectrogram(xb, fs)

# plot spectrogram as sanity check
plt.figure()
plt.imshow(spect)
plt.show(block=False)

# find blockwise peak of spectrogram
maxIndex = np.argmax(spect, axis=0)
# return corresponding frequency vector
f0_fft =  freq[maxIndex]

# # plot returns
plt.figure()
plt.plot(f0_fft)
#plt.plot(f0_hps)
plt.xlabel('Time [s]')
plt.ylabel('Amplitude (raw)')
plt.title('Estimated Pitch')
plt.legend('FFT','HPS')
plt.show()





# f0_hps = track_pitch_hps(x, blockSize, hopSize, fs)
# xb, timeInSec = block_audio(x, blockSize, hopSize, fs)

# # plot returns
# plt.plot(f0_fft)
# plt.plot(f0_hps)
# plt.xlabel('Time [s]')
# plt.ylabel('Amplitude (raw)')
# plt.title('Estimated Pitch')
# plt.legend('FFT','HPS')
# plt.show()

# # calculate absolute error per block
# annotation = np.append(np.ones(np.shape(timeInSec)*f1, np.ones(np.shape(timeInSec*f2))))
# err_fft = np.abs(f0_fft - annotation)
# err_hps = np.abs(f0_hps - annotation)

# # plot errors
# plt.plot(err_fft)
# plt.plot(err_hps)
# plt.xlabel('Time [s]')
# plt.ylabel('Amplitude (raw)')
# plt.title('Error')
# plt.legend('FFT','HPS')
# plt.show()