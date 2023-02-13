import logging
from logging import handlers

import os
import cmapy
import cv2
import librosa
import librosa.display
import numpy as np
from scipy.signal import butter, lfilter
from glob import glob
import random
from tqdm import tqdm
from pesq import pesq


noisy_data_dir_raw = "kumc_lung_clinical_study/noisy/"
clean_data_dir_raw = "kumc_lung_clinical_study/clean/"
target_dir = "./processed/"
sample_rate = 8000
desired_length = 3
lf = 10
hf = 2000





def butter_bandpass(lowcut, highcut, fs, order=5):
    """

    :param lowcut:
    :param highcut:
    :param fs:
    :param order:
    :return:
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, fs, lowcut=100, highcut=2000, order=5):
    """

    :param data:
    :param fs:
    :param lowcut:
    :param highcut:
    :param order:
    :return:
    """
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def create_mel_raw(current_window, sample_rate, n_mels=128, f_min=50, f_max=4000, nfft=2048, hop=512, resz=1):
    """

    :param current_window:
    :param sample_rate:
    :param n_mels:
    :param f_min:
    :param f_max:
    :param nfft:
    :param hop:
    :param resz:
    :return:
    """
    S = librosa.feature.melspectrogram(y=current_window, sr=sample_rate, n_mels=n_mels, fmin=f_min, fmax=f_max,
                                       n_fft=nfft, hop_length=hop)
    S = librosa.power_to_db(S, ref=np.max)
    S = (S - S.min()) / (S.max() - S.min())
    S *= 255
    img = cv2.applyColorMap(S.astype(np.uint8), cmapy.cmap('magma'))
    height, width, _ = img.shape
    if resz > 0:
        img = cv2.resize(img, (width * resz, height * resz), interpolation=cv2.INTER_LINEAR)
    img = cv2.flip(img, 0)
    return img


def generate_padded_samples(original, source, output_length, sample_rate, types):
    """

    :param original:
    :param source:
    :param output_length:
    :param sample_rate:
    :param types:
    :return:
    """
    copy = np.zeros(output_length, dtype=np.float32)
    src_length = len(source)
    left = output_length - src_length  # amount to be padded
    # pad front or back
    prob = random.random()

    aug = original

    while len(aug) < left:
        aug = np.concatenate([aug, aug])

    if prob < 0.5:
        # pad back
        copy[left:] = source
        copy[:left] = aug[len(aug) - left:]
    else:
        # pad front
        copy[:src_length] = source[:]
        copy[src_length:] = aug[:left]

    return copy


def split_and_pad(original, desired_length, sample_rate, types=0):
    """

    :param original:
    :param desired_length:
    :param sample_rate:
    :param types:
    :return:
    """
    output_buffer_length = int(desired_length * sample_rate)
    sound_clip = original[0].copy()
    n_samples = len(sound_clip)

    output = []
    # if: the audio sample length > desiredLength, then split & pad
    # else: simply pad according to given type 1 or 2
    if n_samples > output_buffer_length:
        frames = librosa.util.frame(sound_clip, frame_length=output_buffer_length, hop_length=output_buffer_length // 2,
                                    axis=0)
        for i in range(frames.shape[0]):
            output.append((frames[i]))

        last_id = frames.shape[0] * (output_buffer_length // 2)
        last_sample = sound_clip[last_id:];
        pad_times = (output_buffer_length - len(last_sample)) / len(last_sample)
        padded = generate_padded_samples(sound_clip, last_sample, output_buffer_length, sample_rate, types)
        output.append(padded)

    else:
        padded = generate_padded_samples(sound_clip, sound_clip, output_buffer_length, sample_rate, types);
        pad_times = (output_buffer_length - len(sound_clip)) / len(sound_clip)
        output.append(padded)
    return output


def wav2npy(wavefiles, postfix) :
    random.shuffle(wavfiles_raw)

    data_len = len(wavfiles_raw)

    train_data_len = int(data_len * 0.7)
    valid_data_len = train_data_len + int(data_len * 0.2)
    test_data_len = valid_data_len + int(data_len * 0.1)

    train_audio_data = None
    val_audio_data = None
    test_audio_data = None

    for idx, noisy_f in enumerate(tqdm(wavfiles_raw)):

        clean_f = noisy_f.replace("noisy","clean")
        clean_f = clean_f.replace("Noisy", "Clean")

        sample_n, rate_n = librosa.load(noisy_f, sr=sample_rate)
        sample_c, rate_c = librosa.load(clean_f, sr=sample_rate)

        #sample = butter_bandpass_filter(sample, fs=rate, lowcut=lf, highcut=hf)

        split_data_n = split_and_pad([sample_n], desired_length, sample_rate)
        split_data_c = split_and_pad([sample_c], desired_length, sample_rate)

        # for i in range(len(split_data)-1,0,-1):
        #     wav_data = split_data[i]
        #     clean_wav = wav_data.astype(np.double)
        #
        #     try:
        #         pesq(ref=clean_wav, deg=clean_wav, mode='nb', fs=sample_rate)
        #     except Exception as e:
        #         print(f"{e}")
        #         del split_data[i]

        input_n = np.array(split_data_n)
        target_c = np.array(split_data_c)
        input_n = np.expand_dims(input_n, axis=1)
        target_c = np.expand_dims(target_c, axis=1)
        input_target = np.append(input_n, target_c, axis=1)


        if idx < train_data_len:
            if train_audio_data is not None:
                train_audio_data = np.concatenate((train_audio_data, input_target), axis=0)
            else:
                train_audio_data = input_target
        elif idx < valid_data_len:
            if val_audio_data is not None:
                val_audio_data = np.concatenate((val_audio_data, input_target), axis=0)
            else:
                val_audio_data = input_target
        else:
            if test_audio_data is not None:
                test_audio_data = np.concatenate((test_audio_data, input_target), axis=0)
            else:
                test_audio_data = input_target



    print(train_audio_data.shape)
    print(val_audio_data.shape)
    print(test_audio_data.shape)

    np.save(f'./train.npy', train_audio_data)
    np.save(f'./val.npy', val_audio_data)
    np.save(f'./test.npy', test_audio_data)



random.seed(20221031)
wavfiles_raw = glob(noisy_data_dir_raw + '*.wav')
wav2npy(wavfiles_raw, "noisy")



