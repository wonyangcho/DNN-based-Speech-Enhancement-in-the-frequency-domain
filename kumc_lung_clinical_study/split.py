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


source_data_dir_raw = "raw/audio_txt_files/"
noise_data_dir_raw = "noise/"
target_dir = "./processed/"
sample_rate = 8000
desired_length = 7
lf = 10
hf = 2000


def initLogger():
    """

    :return:
    """
    global myLogger
    # 현재 파일 경로 및 파일명 찾기
    current_dir = os.path.dirname(os.path.realpath(__file__))
    current_file = os.path.basename(__file__)
    current_file_name = current_file[:-3]  # xxxx.py
    LOG_FILENAME = 'log-{}'.format(current_file_name)

    # 로그 저장할 폴더 생성
    log_dir = '{}/logs'.format(current_dir)
    os.makedirs(log_dir, exist_ok=True)

    # 로거 생성
    LOG_FILENAME = "skeeperweb.log"
    MY_LOG_FILE_PATH = f"{log_dir}/{LOG_FILENAME}"
    logging.basicConfig(filename=MY_LOG_FILE_PATH)
    myLogger = logging.getLogger(LOG_FILENAME)  # 로거 이름: test
    myLogger.setLevel(logging.DEBUG)  # 로깅 수준: DEBUG

    LOG_FILE_MAX_BYTES = 50 * 1024 * 1024  # 50MB

    file_handler = handlers.RotatingFileHandler(MY_LOG_FILE_PATH, maxBytes=LOG_FILE_MAX_BYTES, backupCount=10)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] %(message)s')
    file_handler.setFormatter(formatter)  # 핸들러에 로깅 포맷 할당
    myLogger.addHandler(file_handler)  # 로거에 핸들러 추가
    myLogger.propagate = False


def error_log(level, error_str):
    """

    :param level:
    :param error_str:
    :return:
    """
    global myLogger

    if level == logging.INFO:
        myLogger.info("{}".format(error_str))
    if level == logging.DEBUG:
        myLogger.debug("{}".format(error_str))
    if level == logging.ERROR:
        myLogger.error("{}".format(error_str))


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


wavfiles_raw = glob(source_data_dir_raw + '/*/*.wav')
wavfiles_raw.extend(glob(source_data_dir_raw + '/*/*/*.wav'))
wavfiles_raw.extend(glob(source_data_dir_raw + '/*.wav'))
wavfiles_raw.extend(glob(noise_data_dir_raw + '/*.wav'))

random.seed(20221031)
random.shuffle(wavfiles_raw)

data_len = len(wavfiles_raw)

train_data_len = int(data_len*0.7)
valid_data_len = train_data_len+int(data_len*0.2)
test_data_len = valid_data_len+int(data_len*0.1)

train_audio_data = None
val_audio_data = None
test_audio_data = None

for idx, f in enumerate(tqdm(wavfiles_raw)):
    sample, rate = librosa.load(f, sr=sample_rate)
    sample = butter_bandpass_filter(sample, fs=rate, lowcut=lf, highcut=hf)
    split_data = split_and_pad([sample], desired_length, sample_rate)


    for i in range(len(split_data)-1,0,-1):
        wav_data = split_data[i]
        clean_wav = wav_data.astype(np.double)

        try:
            pesq(ref=clean_wav, deg=clean_wav, mode='nb', fs=sample_rate)
        except Exception as e:
            print(f"{e}")
            del split_data[i]


    input_target = np.array(split_data)
    input_target = np.expand_dims(input_target, axis=1)
    input_target = np.append(input_target,input_target,axis=1)

    if idx < train_data_len:
        if train_audio_data is not None:
            train_audio_data = np.concatenate((train_audio_data,input_target),axis=0)
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

np.save('./train.npy', train_audio_data)
np.save('./val.npy', val_audio_data)
np.save('./test.npy', test_audio_data)
