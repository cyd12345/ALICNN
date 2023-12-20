import pysepm
import numpy as np
import wave
import pyaudio
import matplotlib.pyplot as plt
import librosa
import logging
import math

def audio_assessment(scorer_,audio1,audio2,samplerate):
    Scorer = {'pesq':pysepm.pesq,
              'stoi':pysepm.stoi,
              'SNR':pysepm.SNRseg}
    scorer = Scorer[scorer_]
    if audio1.ndim == 1:
        audio1 = np.expand_dims(audio1,axis=0)
        audio2 = np.expand_dims(audio2, axis=0)
    batch_size = audio1.shape[0]
    Score = 0
    error_num = 0
    for i in np.arange(0,batch_size):
        try:
            if len(audio1[i])==len(audio2[i]):
                score_ = scorer(audio1[i],audio2[i],samplerate)
            else:
                lenmin = min(len(audio1[i]),len(audio2[i]))
                audio1 = audio1[i][:lenmin]
                audio2 = audio2[i][:lenmin]
                score_ = scorer(audio1[i],audio2[i],samplerate)
            if isinstance(score_,tuple):
                score_ = score_[-1]
            Score += score_
        except:
            error_num += error_num
            print('error')

    Score /= (batch_size - error_num)
    return Score

def audiostoi(audio1,audio2,samplerate):
    if len(audio1)==len(audio2):
        score = pysepm.stoi(audio1,audio2,samplerate)
    else:
        lenmin = min(len(audio1),len(audio2))
        audio1 = audio1[:lenmin]
        audio2 = audio2[:lenmin]
        score = pysepm.stoi(audio1,audio2,samplerate)
    return score

def audioSNR(audio1,audio2,samplerate):
    if len(audio1)==len(audio2):
        score = pysepm.SNRseg(audio1,audio2,samplerate)
    else:
        lenmin = min(len(audio1),len(audio2))
        audio1 = audio1[:lenmin]
        audio2 = audio2[:lenmin]
        score = pysepm.SNRseg(audio1,audio2,samplerate)
    return score

def audiorecover(tf,phase,mean, savepath=None,save=True):
    fftArray = np.multiply(tf, np.exp(phase * 1j))
    isou = librosa.istft(fftArray, hop_length=256, win_length=1024, window='hamming', center=True, length=None)
    isou = isou*(mean[0]/(np.mean(np.abs(isou))))

    # Because mixed = clean + SNR_factor*noise, rescale to range(-1,1)
    #isou = np.divide(isou, np.max(abs(isou)))

    # Scale up
    #recovered = scaleUp(isou, N)

    if save:
                 record(isou,savepath)

    return isou

def savemap(response_attention_map,response_based_sum_c_maps,savepath):
    plt.figure(figsize=(10, 10))
    # plt.subplot(6, 1, 6)
    # librosa.display.specshow(response_attention_map, y_axis='linear')
    # plt.colorbar()
    # plt.title('sum')
    for i in np.arange(0, 5):
        plt.subplot(6, 1, i + 1)
        librosa.display.specshow(response_based_sum_c_maps[i], y_axis='linear')
        plt.colorbar()
        plt.title(i + 1)
    plt.savefig(savepath)

def savemask(maps,savepath):
    """
        Function prints three maps for each ReLu layer: suppression mask, ReLu activation map before lateral inhibition
          and ReLu layer activation after lateral inhibition.
    """
    pnum = 1
    relu_layers = ['encoder1', 'encoder2', 'encoder3', 'encoder4', 'encoder5']

    for l in relu_layers:
            plt.subplot(5, 3, pnum)
            pnum += 1
            plt.imshow(maps[l][0], cmap='viridis')
            bar = plt.colorbar(shrink=1.0)
            bar.ax.tick_params(labelsize=4)
            plt.tick_params(labelsize=6)
            plt.title(f"suppression mask - {l}",fontsize='small')

            plt.subplot(5, 3, pnum)
            pnum += 1
            plt.imshow(maps[l][1], cmap='viridis')
            bar = plt.colorbar(shrink=1.0)
            bar.ax.tick_params(labelsize=4)
            plt.tick_params(labelsize=6)
            plt.title(f"normal ReLu - {l}",fontsize='small')

            plt.subplot(5, 3, pnum)
            pnum += 1
            plt.imshow(maps[l][2], cmap='viridis')
            bar = plt.colorbar(shrink=1.0)
            bar.ax.tick_params(labelsize=4)
            plt.tick_params(labelsize=6)
            plt.title(f"inhibited ReLu - {l}",fontsize='small')
    plt.savefig(savepath)

def savedb(target,clr,mask,pro,savepath):
    clr = librosa.amplitude_to_db(clr, ref=np.min)
    pro = librosa.amplitude_to_db(pro, ref=np.min)
    plt.figure(figsize=(8, 10))
    plt.subplot(4, 1, 1)
    librosa.display.specshow(target, y_axis='linear')
    plt.colorbar()
    plt.title('target IRM')
    plt.subplot(4, 1, 3)
    librosa.display.specshow(clr, y_axis='linear')
    plt.colorbar(format='%+2.0f dB')
    plt.title('clear spectrogram')
    plt.subplot(4, 1, 2)
    librosa.display.specshow(mask, y_axis='linear')
    plt.colorbar()
    plt.title('processed IRM')
    plt.subplot(4, 1, 4)
    librosa.display.specshow(pro, y_axis='linear')
    plt.colorbar(format='%+2.0f dB')
    plt.title('processed spectrogram')
    plt.savefig(savepath)

def scaleUp(a,N):
    """ Scale up from  [-1,1] to intN,
    only int16 is used.

    # Arguments
        N: int type.
        a: vector

    # Returns
        Upscaled vector
    """


    b = list(map(lambda x : x*(2**(N-1)-1),a))
    if N == 32:
        return np.array(b,dtype= np.int32)
    if N==16:
        return np.array(b,dtype = np.int16)
    return 0

def overlapAdd(windowArray):
    """ Add the windowed signal to reconstruct the original signal sequence.
    Help function used in recoverSignal and recoverSignalStandard.


    # Arguments
        windowArray: Nwindows x windowLength matrix, where the windows are overlapping with 50 %

    # Returns
        overlapped: vector with overlap-added signal
    """

    [Nwindows, windowLength] = windowArray.shape
    Ntot = int(Nwindows * (windowLength / 4)+windowLength)
    overlapped = np.zeros(Ntot)
    steplen = int(windowLength / 4)
    # Add the first half window manually
    for i in range(0, Nwindows):
        # Add the elements corresponding to the current half window
        overlapped[steplen*i:windowLength+steplen*i] += windowArray[i,:]

    return overlapped

p = pyaudio.PyAudio()

def record(re_frames, WAVE_OUTPUT_FILENAME,CHANNELS = 1,FORMAT = pyaudio.paInt16,RATE = 16000):
    re_frames = np.real(re_frames)
    re_frames = ((re_frames - re_frames.mean()) / (re_frames.max()))*6000
    re_frames = re_frames.astype(np.short).tobytes()
    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(re_frames)
    wf.close()

def train_logger(filename,lgname='trainlogger',lglevel=logging.INFO):
    trainlogger = logging.Logger(lgname)
    trainhandler = logging.FileHandler(filename)
    trainhandler.setLevel(lglevel)
    trainformat = logging.Formatter("time:%(asctime)s message:%(message)s line:%(lineno)d")
    trainhandler.setFormatter(trainformat)
    trainlogger.addHandler(trainhandler)
    return trainlogger

def SNR_singlech(S, SN):
    S = S-np.mean(S)# 消除直流分量
    S = S/np.max(np.abs(S))#幅值归一化
    mean_S = (np.sum(S))/(len(S))#纯信号的平均值
    PS = np.sum((S-mean_S)*(S-mean_S))
    PN = np.sum((S-SN)*(S-SN))
    snr=10*math.log((PS/PN), 10)
    return(snr)