import os
import shutil
import torchaudio
import torch
import numpy as np


def mix_2_with_specified_snr(wav_tensor_1, wav_tensor_2, snr_ratio):
    power_1 = torch.sqrt(torch.sum(wav_tensor_1 ** 2))
    power_2 = torch.sqrt(torch.sum(wav_tensor_2 ** 2))
    new_power_ratio = np.sqrt(np.power(10., snr_ratio / 10.))
    new_wav_tensor_1 = new_power_ratio * wav_tensor_1 / (power_1 + 10e-8)
    new_wav_tensor_2 = wav_tensor_2 / (power_2 + 10e-8)
    return new_wav_tensor_1, new_wav_tensor_2


# 경로 설정
clean_data_dir = '/work/wycho/project/DNN-based-Speech-Enhancement-in-the-frequency-domain/data/wav/'
noise_data_dir = '/work/wycho/project/DNN-based-Speech-Enhancement-in-the-frequency-domain/data/noise/'
mixture_data_dir = '/work/wycho/project/DNN-based-Speech-Enhancement-in-the-frequency-domain/data/noisy'

snr_ratios = [-10, -5, 0, 5, 10]

# 폴더 생성 함수


def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)

# 데이터 섞기


def mix_data(clean_dir, noise_dir, mixture_dir, snr_ratios):
    # Clean data
    create_directory(mixture_dir)
    for snr_ratio in snr_ratios:
        snr_dir = os.path.join(mixture_dir, str(snr_ratio) + "db")
        create_directory(snr_dir)
        for split in ['train', 'val', 'test']:
            clean_split_dir = os.path.join(clean_dir, split)
            noise_split_dir = os.path.join(noise_dir, split)
            mixture_split_dir = os.path.join(snr_dir, split)
            create_directory(mixture_split_dir)

            clean_files = os.listdir(clean_split_dir)
            noise_files = os.listdir(noise_split_dir)

            for i, clean_file in enumerate(clean_files):
                clean_path = os.path.join(clean_split_dir, clean_file)
                noise_file = noise_files[i % len(noise_files)]
                noise_path = os.path.join(noise_split_dir, noise_file)
                # mixture_file = f'mixture_{snr_ratio}db_{i+1}.wav'
                mixture_file = os.path.basename(clean_path)
                mixture_path = os.path.join(mixture_split_dir, mixture_file)

                # Load audio files at 8kHz
                clean_waveform, sr_clean = torchaudio.load(clean_path)
                noise_waveform, sr_noise = torchaudio.load(noise_path)

                # print(f"{sr_clean} {sr_noise}")
                # Mix audio with specified SNR ratio
                mixed_waveform_1, mixed_waveform_2 = mix_2_with_specified_snr(
                    clean_waveform, noise_waveform, snr_ratio)

                # Resample mixed waveform to 16kHz and save as WAV file
                mixed_waveform_16k = torchaudio.transforms.Resample(
                    8000, 16000)(mixed_waveform_1)
                torchaudio.save(
                    mixture_path, mixed_waveform_16k, sample_rate=16000)


# 데이터 섞기 함수 호출
mix_data(clean_data_dir, noise_data_dir, mixture_data_dir, snr_ratios)
