import wave
import numpy as np
import scipy.signal as signal
from scipy.fftpack import fft
import librosa
import librosa.display
import os

def libfft(filename,mintime,nw=1024,inc=256):
    sou, sr = librosa.load(filename, sr=None, mono=True, offset=0.0, duration=mintime)
    tf = librosa.stft(sou, n_fft=nw, hop_length=inc, win_length=nw, window='hamming', center=True,
                      pad_mode='reflect')
    stft_matrix = np.abs(tf)
    stft_matrix /= stft_matrix.max()
    phase = np.angle(tf)

    #im = Image.fromarray(stft_matrix/stft_matrix.max() * 255)
    #im = im.convert('L')
    #im.save(r'/home/cyd/speech_test/LICNN_SPEECH/{}/FFTMAP/{}-{}.jpg'.format(project, name, type))
    return sou, stft_matrix, phase

def fftaudio(filename,mintime,nw=1024,inc=256):
    f = wave.open(filename, 'rb')
    params = f.getparams()
    nchannels, sampwidth, framerate, nframes = params[:4]
    strData = f.readframes(nframes)
    strData = strData[:framerate*4]
    f.close()

    waveData = np.fromstring(strData, dtype=np.int16) #len=len/2
    waveData = waveData * 1.0 / (max(abs(waveData)))
    r'''
    time = np.arange(0, len(strData)) * (1.0 / framerate)
    plt.plot(time, waveData)
    plt.xlabel("Time(s)")
    plt.ylabel("Amplitude")
    plt.title("Single channel wavedata")
    plt.grid('on')  # ������on������off:����
    # plt.savefig(r'D:\Speech_Separation_pytorch\SINGLE1\{}.jpg'.format(name))
    plt.cla()
    '''
    winfunc = signal.hamming(nw)
    Frame = enframe(waveData, nw, inc, winfunc)

    fft_signal = fft(Frame)
    signalfft = np.abs(fft_signal)
    shape = signalfft.shape
    signalfft = signalfft.reshape(1, -1)
    signalfft = signalfft / signalfft.max()
    signalfft = signalfft.reshape(shape)

    signalfft = signalfft[:, -(nw//2+1):]
    signalfft = signalfft.transpose(1, 0)
    phase = np.angle(fft_signal)
    phase = phase[:, -(nw//2+1):]
    phase = phase.transpose(1,0)
    return waveData,signalfft,phase


def enframe(signal, nw, inc, winfunc=None):
    signal_length = len(signal)  # ����������
    if signal_length <= nw:  # ����������������������������������������1
        nf = 1
    else:  # ��������������
        nf = int(np.ceil((1.0 * signal_length - nw + inc) / inc))

    pad_length = int((nf - 1) * inc + nw)  # ����������������������������
    zeros = np.zeros((pad_length - signal_length))  # ��������������0������������FFT����������������
    pad_signal = np.concatenate((signal, zeros))  # ����������������pad_signal
    indices = np.tile(np.arange(0, nw), (nf, 1)) + np.tile(np.arange(0, nf * inc, inc), (nw, 1)).T
    indices = np.array(indices, dtype=np.int32)  # ��indices����������
    frames = pad_signal[indices]  # ����������
    if winfunc:
        win = np.tile(winfunc, (nf, 1))  # window������������������1
        return frames * win  # ��������������
    return frames

def mkdir(path):
    folder = os.path.exists(path)

    if not folder:
        os.makedirs(path)

def signal1D_to_matrix2D(filename,mintime,nw=320,inc=160):
    sou, sr = librosa.load(filename, sr=None, mono=True, offset=0.0, duration=mintime)
    matrix = enframe(sou,nw,inc)
    matrix = matrix.T
    return matrix

def matrix2D_to_sighal1D(matrix,inc=160):
    nw = matrix.shape[0]
    nf = matrix.shape[1]
    matrix = matrix.T
    matrix[1:,:inc] = (matrix[1:,:inc]+matrix[:-1,nw-inc:])/2
    signal = np.append(matrix[:,:nw-inc].reshape(nf*(nw-inc)),matrix[-1,-inc:])
    return signal