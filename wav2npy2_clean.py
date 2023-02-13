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
import scipy.io.wavfile as wav
from tqdm import tqdm
#from pesq import pesq


noisy_data_dir_raw = "noisy/"
clean_data_dir_raw = "clean/"
target_dir = "./processed/"
sample_rate = 8000
desired_length = 3
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
                                    axis=0) ###
        for i in range(frames.shape[0]):
            output.append((frames[i]))

        last_id = frames.shape[0] * (output_buffer_length // 2)
        last_sample = sound_clip[last_id:]
        pad_times = (output_buffer_length - len(last_sample)) / len(last_sample)
        padded = generate_padded_samples(sound_clip, last_sample, output_buffer_length, sample_rate, types)
        output.append(padded)

    else:
        padded = generate_padded_samples(sound_clip, sound_clip, output_buffer_length, sample_rate, types);
        pad_times = (output_buffer_length - len(sound_clip)) / len(sound_clip)
        output.append(padded)
    return output

def generate_noisy_wav(clean_temp, noise_wav, snr_in_db):
    # speech 및 noise 성분의 길이를 구합니다.
    # speech 신호와 길이가 같도록 noise segmnet를 무작위로 선택합니다.
    # DC 바이어스를 제거한 후 speech 및 noise의 power을 계산합니다.
    #clean_temp, sr = librosa.load(clean_wav)
    #print(clean_temp.shape)
    randNoisy, noise_sr = librosa.load(noise_wav, sr=8000)
    randNoisy = librosa.resample(randNoisy, noise_sr, sample_rate)
    dc_speech = np.mean(clean_temp)
    dc_noise = np.mean(randNoisy)
    pow_speech = np.mean((clean_temp - dc_speech)**2)
    pow_noise = np.mean((randNoisy - dc_noise)**2)
    # target SNR에 따라 노이즈 성분의 scale factor를 계산합니다.
    alpha = np.sqrt(10.0 ** (float(-snr_in_db) / 10.0) * pow_speech / (pow_noise + 1e-6))
    #print(clean_temp.shape) # 1323000
    #print(randNoisy.shape) # 5186831
    return clean_temp + (randNoisy[:clean_temp.shape[0]] * alpha), sample_rate

def generate_lung_noisy_wav(cleanWav, NoiseWav, snr_in_db, artNum, len_lungWav):
    snr_set = [-5, 0, 5, 10]
    print(NoiseWav)
    NoiseWav, noise_sr = librosa.load(NoiseWav, sr=8000)
    print(NoiseWav.shape)
    len_artiWav = len(NoiseWav)

    dc_speech = np.mean(cleanWav)
    dc_noise = np.mean(NoiseWav) 
    pow_speech = np.mean((cleanWav - dc_speech)**2)
    pow_noise = np.mean((NoiseWav - dc_noise)**2)

    subname = ''
    for i in range(artNum): # Random artifact Num
        # Compute the scale factor of noise component depending on the target SNR.
        snrIndex = np.random.randint(0, len(snr_set)-1)
        snr = snr_set[snrIndex]
        alpha = np.sqrt(10.0 ** (float(-snr) / 10.0) * pow_speech / (pow_noise + 1e-6))
        #print(alpha) # 0.5890878061050928

        if len_artiWav < len_lungWav:
            st = np.random.randint(0, len_lungWav - len_artiWav) # random start position
            ed = st + len_artiWav
            cleanWav[st:ed] = cleanWav[st:ed] + (NoiseWav * alpha)
        else:
            cleanWav = (NoiseWav[:len(cleanWav)] * alpha) + cleanWav
        subname += str(snr) + '_'

    # target SNR에 따라 노이즈 성분의 scale factor를 계산합니다.
    alpha = np.sqrt(10.0 ** (float(-snr_in_db) / 10.0) * pow_speech / (pow_noise + 1e-6))

    if NoiseWav.shape[0] > cleanWav.shape[0]:
        ans = (NoiseWav[:cleanWav.shape[0]] * alpha) + cleanWav
    else:
        ans = NoiseWav + cleanWav[:len(NoiseWav)]
    return ans, noise_sr, subname

def wav2npy(lungWav, artiWav, dir_to_save) :
    random.shuffle(lungWav)
    data_len = len(lungWav)

    train_data_len = int(data_len * 0.8)
    valid_data_len = train_data_len + int(data_len * 0.1)
    test_data_len = valid_data_len + int(data_len * 0.1)

    train_audio_data = None
    val_audio_data = None
    test_audio_data = None

    for idx, lung in enumerate(tqdm(lungWav)): # clean wav
        snr_set = [-5, 0, 5, 10]
        snr_idx = np.random.randint(0, len(snr_set))
        snr_in_db = snr_set[snr_idx] # random snr choice

        snr_idx = snr_idx + 1

        random_idx = np.random.randint(0, len(artiWav)) # artifact(noise) random choice
        sample_c, rate_c = librosa.load(lung, sr=8000) # clean wav read
        len_artiWav = len(artiWav)
        len_lungWav = len(sample_c)
        sample_n, rate_n, subname = generate_lung_noisy_wav(sample_c, artiWav[random_idx], snr_in_db, snr_idx, len_lungWav) # random snr noisy wav

        noisy_name = lung.split('/')[-1].split('.')[0] + artiWav[random_idx].split('/')[-1].split('.')[0] + '_snr' + subname
        total_path = dir_to_save + '/' + noisy_name + '.wav'
        clean_path = dir_to_save + '/' + lung.split('/')[-1]
        if sample_c.shape[0] > sample_n.shape[0]:
            sample_c = sample_c[:sample_n.shape[0]]
        else:
            sample_n = sample_n[:sample_c.shape[0]]

        split_data_n = split_and_pad([sample_n], desired_length, sample_rate) # noisy siganl, desired_length, smpl
        split_data_c = split_and_pad([sample_c], desired_length, sample_rate)

        input_n = np.array(split_data_n)
        target_c = np.array(split_data_c)
        input_n = np.expand_dims(input_n, axis=1)
        target_c = np.expand_dims(target_c, axis=1)
        input_target = np.append(input_n, target_c, axis=1)

        clean_target = np.append(target_c, target_c, axis=1)


        if idx < train_data_len:
            wav.write(total_path, 8000, sample_n)
            wav.write(clean_path, 8000, sample_c)
            if train_audio_data is not None:
                train_audio_data = np.concatenate((train_audio_data, input_target), axis=0)
                train_audio_data = np.concatenate((train_audio_data, clean_target), axis=0)
            else:
                train_audio_data = input_target
        elif idx < valid_data_len:
            wav.write(total_path.replace('train','valid'), 8000, sample_n)
            wav.write(clean_path.replace('train','valid'), 8000, sample_c)
            if val_audio_data is not None:
                val_audio_data = np.concatenate((val_audio_data, input_target), axis=0)
                val_audio_data = np.concatenate((val_audio_data, clean_target), axis=0)
            else:
                val_audio_data = input_target
        else:
            wav.write(total_path.replace('train','test'), 8000, sample_n)
            wav.write(clean_path.replace('train','test'), 8000, sample_c)
            if test_audio_data is not None:
                test_audio_data = np.concatenate((test_audio_data, input_target), axis=0)
                test_audio_data = np.concatenate((test_audio_data, clean_target), axis=0)
            else:
                test_audio_data = input_target


    print(train_audio_data.shape)
    print(val_audio_data.shape)
    print(test_audio_data.shape)
    
    # # real data
    # np.save(f'/work/hyerim/dccrn/data_json/realnoise_train.npy', train_audio_data)
    # np.save(f'/work/hyerim/dccrn/data_json/realnoise_val.npy', val_audio_data)
    # np.save(f'/work/hyerim/dccrn/data_json/realnoise_test.npy', test_audio_data)
     
    # noisex92 data
    np.save(f'/work/hyerim/NoiseNew/dataset/lung_addclean_train.npy', train_audio_data)
    np.save(f'/work/hyerim/NoiseNew/dataset/lung_addclean_val.npy', val_audio_data)
    np.save(f'/work/hyerim/NoiseNew/dataset/lung_addclean_test.npy', test_audio_data)


def main():

    random.seed(20221031)
    artiWav = glob('/work/hyerim/NoiseNew/split_dataset/artifact/app_filtered/*.wav') # artifact
    crackleWav = glob('/work/hyerim/NoiseNew/split_dataset/crackle/app_filtered/*.wav') # crackle
    wheezWav = glob('/work/hyerim/NoiseNew/split_dataset/wheeze/app_filtered/*.wav') # wheeze
    normWav = glob('/work/hyerim/NoiseNew/split_dataset/wheeze/app_filtered/*.wav') # normal

    lungWav = wheezWav + normWav + crackleWav

    dir_to_save = '/work/hyerim/NoiseNew/dataset/addclean_train'

    # make the file directory
    if not os.path.exists(dir_to_save):
        os.mkdir(dir_to_save)
    dir_to_save_valid = dir_to_save.replace('train','valid')
    if not os.path.exists(dir_to_save_valid):
        os.mkdir(dir_to_save_valid)
    dir_to_save_test = dir_to_save.replace('train','test')
    if not os.path.exists(dir_to_save_test):
        os.mkdir(dir_to_save_test)

    wav2npy(lungWav, artiWav, dir_to_save)

if __name__ == "__main__":
    main()
