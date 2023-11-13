## First module for Assignment A4 - Creates functions for Tuning Frequency Estimation

# test
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
def block_audio(x, blockSize, hopSize, fs):
    # allocate memory
    numBlocks = math.ceil(x.size / hopSize)
    xb = np.zeros([numBlocks, blockSize])
    # compute time stamps
    t = (np.arange(0, numBlocks) * hopSize) / fs
    x = np.concatenate((x, np.zeros(blockSize)), axis=0)
    for n in range(0, numBlocks):
        i_start = n * hopSize
        i_stop = np.min([x.size - 1, i_start + blockSize - 1])
        xb[n][np.arange(0, blockSize)] = x[np.arange(i_start, i_stop + 1)]
    return (xb, t)


def compute_hann(iWindowLength):
    return 0.5 - (0.5 * np.cos(2 * np.pi / iWindowLength * np.arange(iWindowLength)))


def compute_spectrogram(xb, fs):
    numBlocks = xb.shape[0]
    afWindow = compute_hann(xb.shape[1])
    X = np.zeros([math.ceil(xb.shape[1] / 2 + 1), numBlocks])
    for n in range(0, numBlocks):
        # apply window
        tmp = abs(sp.fft.fft(xb[n, :] * afWindow)) * 2 / xb.shape[1]
        # compute magnitude spectrum
        X[:, n] = tmp[range(math.ceil(tmp.size / 2 + 1))]
        X[[0, math.ceil(tmp.size / 2)], n] = X[
            [0, math.ceil(tmp.size / 2)], n
        ] / np.sqrt(2)
    # let's be pedantic about normalization
    f = np.arange(0, X.shape[0]) * fs / (xb.shape[1])
    return (X, f)


def convert_freq2midi(fInHz, fA4InHz=440):
    def convert_freq2midi_scalar(f, fA4InHz):
        if f.any() <= 0:
            return 0
        else:
            return 69 + 12 * np.log2(f / fA4InHz)

    fInHz = np.asarray(fInHz)
    if fInHz.ndim == 0:
        return convert_freq2midi_scalar(fInHz, fA4InHz)
    midi = np.zeros(fInHz.shape)
    for k, f in enumerate(fInHz):
        midi[k] = convert_freq2midi_scalar(f, fA4InHz)
    return midi


def get_spectral_peaks(X):
    """
    Returns the 20 largest spectral peaks of a signal - for a given column X of a spectrogram
    """
    # get the 20 largest peaks
    peakinds, _ = sp.signal.find_peaks(
        X, distance=1, height=None
    )  # adjust distance and height for avoiding neighboring frequencies - depends on spectral resolution, to adjust
    # sort the peaks by magnitude
    peakinds = peakinds[np.argsort(X[peakinds])][::-1]
    # return the 20 largest peaks
    peaks = peakinds[:20]
    return peaks


def estimate_tuning_freq(x, blockSize, hopSize, fs):
    # block signal
    xb, timeInSec = block_audio(
        x, blockSize, hopSize, fs
    )  # imported from A1 helper module
    X, fInHz = compute_spectrogram(xb, fs)  # imported from A3 module

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
    # plt.figure()
    # plt.plot(bin_edges[1:], hist)
    # plt.xlabel("Deviation (cents)")
    # plt.ylabel("Count")
    # plt.title("Histogram of Deviation")
    # plt.show()
    # return the bin with the highest count
    tuningFreq = bin_edges[maxbin]
    # convert the tuning frequency to Hz
    tuningFreq = 440 * 2 ** (tuningFreq / 1200)
    # return the tuning frequency
    return tuningFreq


def extract_pitch_chroma(X, fs, tfInHz):
    n_bins, numBlocks = X.shape
    pitchChroma = np.zeros((12, numBlocks))
    # frequency range from C3 to B5
    lowerBound = 130.81
    upperBound = 987.77
    frequencies = np.linspace(0, fs / 2, n_bins)
    for i in range(numBlocks):
        chroma = np.zeros(12)
        for j in range(1, n_bins):
            if frequencies[j] < lowerBound:
                continue
            if frequencies[j] > upperBound:
                break
            pitch_class = int(np.round(12 * np.log2(frequencies[j] / tfInHz))) % 12
            chroma[pitch_class] += X[j, i]
        norm = np.linalg.norm(chroma, ord=2)
        if norm > 0:
            chroma /= norm
        pitchChroma[:, i] = chroma
    return pitchChroma


def detect_key(x, blockSize, hopSize, fs, bTune):
    t_pc = np.array(
        [
            [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88],
            [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17],
        ]
    )

    t_major = t_pc[0] / np.linalg.norm(t_pc[0])
    t_minor = t_pc[1] / np.linalg.norm(t_pc[1])

    xb, timeInSec = block_audio(x, blockSize, hopSize, fs)
    X, fInHz = compute_spectrogram(xb, fs)

    if bTune:
        tfInHz = estimate_tuning_freq(x, blockSize, hopSize, fs)
    else:
        tfInHz = 440

    pitchChroma = extract_pitch_chroma(X, fs, tfInHz)

    avgChroma = np.mean(pitchChroma, axis=1)
    # normalize
    avgChroma /= np.linalg.norm(avgChroma)

    distance_major = [
        np.linalg.norm(avgChroma - np.roll(t_major, i)) for i in range(12)
    ]
    distance_minor = [
        np.linalg.norm(avgChroma - np.roll(t_minor, i)) for i in range(12)
    ]

    key_major = np.argmin(distance_major)
    key_minor = np.argmin(distance_minor)

    if distance_major[key_major] < distance_minor[key_minor]:
        key = key_major
    else:
        key = key_minor + 12

    return key


def eval_tfe(pathToAudio, pathToGT):
    block_size = 4096
    hop_size = 2048
    deviation_cent = []
    for audio in os.listdir(pathToAudio):
        fs, x = wav.read(os.path.join(pathToAudio, audio))
        # read txt file
        with open(os.path.join(pathToGT, audio[:-4] + ".txt")) as f:
            tuningFreqGT = float(f.read())
        tuningFreq = estimate_tuning_freq(x, block_size, hop_size, fs)
        deviation_cent.append(1200 * np.log2(tuningFreq / tuningFreqGT))
    deviation_cent = np.array(deviation_cent)
    average_deviation = np.mean(deviation_cent)

    print("Average deviation (cents): ", average_deviation, "cents")
    return average_deviation


def eval_key_detection(pathToAudio, pathToGT):
    block_size = 4096
    hop_size = 2048
    accuracy_tuned = 0
    accuracy = 0
    for audio in os.listdir(pathToAudio):
        fs, x = wav.read(os.path.join(pathToAudio, audio))
        # read txt file
        with open(os.path.join(pathToGT, audio[:-4] + ".txt")) as f:
            keyGT = int(f.read())
        key_tuned = detect_key(x, block_size, hop_size, fs, bTune=True)
        key = detect_key(x, block_size, hop_size, fs, bTune=False)
        if key_tuned == keyGT:
            accuracy_tuned += 1
        if key == keyGT:
            accuracy += 1
        # # print file name
        # print(audio)
        # print("Estimated key (tuned): ", key_tuned)
        # print("Estimated key: ", key)
        # print("Ground truth key: ", keyGT)
        # print("")
    accuracy_tuned /= len(os.listdir(pathToAudio))
    accuracy /= len(os.listdir(pathToAudio))
    print("Accuracy (tuned): ", accuracy_tuned)
    print("Accuracy: ", accuracy)
    return np.array([accuracy_tuned, accuracy])


def valuate(pathToAudioKey, pathToGTKey, pathToAudioTf, pathToGTTf):
    avg_accuracy = eval_key_detection(pathToAudioKey, pathToGTKey)
    avg_deviationInCent = eval_tfe(pathToAudioTf, pathToGTTf)
    return avg_accuracy, avg_deviationInCent


if __name__ == "__main__":
    avg_accuracy, avg_deviationInCent = valuate(
        r"/Users/ljr/Desktop/homework/MUSI-6201-Assignments/key_tf/key_eval/audio",
        r"/Users/ljr/Desktop/homework/MUSI-6201-Assignments/key_tf/key_eval/GT",
        r"/Users/ljr/Desktop/homework/MUSI-6201-Assignments/key_tf/tuning_eval/audio",
        r"/Users/ljr/Desktop/homework/MUSI-6201-Assignments/key_tf/tuning_eval/GT",
    )
    print("Average accuracy: ", avg_accuracy)
    print("Average deviation (cents): ", avg_deviationInCent)
