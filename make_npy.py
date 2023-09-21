import os
import numpy as np
import torchaudio
import glob

# 오디오 데이터를 저장할 목표 샘플 속도 (샘플링 레이트)
TARGET_SR = 8000


def save_data_as_npy(input_dirs, target_dir, output_dir):
    for split in ['train', 'val', 'test']:
        input_paths = []
        target_paths = []

        # 입력 디렉토리에서 오디오 파일 경로 수집
        for input_dir in input_dirs:
            split_input_dir = os.path.join(input_dir, split)
            input_paths.extend(
                glob.glob(os.path.join(split_input_dir, '*.wav')))
        input_paths.sort()

        # 대상 디렉토리에서 대상 파일 경로 수집
        split_target_dir = os.path.join(target_dir, split)
        target_paths = []
        for input_path in input_paths:
            filename = os.path.basename(input_path)
            target_path = os.path.join(split_target_dir, filename)
            if os.path.exists(target_path):
                target_paths.append(target_path)
            else:
                raise ValueError(f"Target file not found for {filename}")

        # 첫 번째 입력 파일을로드하여 대상 샘플 속도에 맞게 리샘플링
        # numpy 배열의 길이를 계산하기 위해
        input_data, sr_input_data = torchaudio.load(input_paths[0])
        if sr_input_data != TARGET_SR:
            input_data = torchaudio.transforms.Resample(
                sr_input_data, TARGET_SR)(input_data)

        num_data = len(input_paths)
        data_length = len(input_data[0])

        # 데이터를 저장할 NumPy 배열 생성
        data_array = np.empty((num_data, 2, data_length), dtype=np.float32)

        print(data_array.shape)

        # 모든 데이터를 처리하고 NumPy 배열에 저장
        for i in range(num_data):
            print(f"{i} {input_paths[i]} {target_paths[i]}")
            input_waveform, sr_input = torchaudio.load(input_paths[i])
            target_waveform, sr_target = torchaudio.load(target_paths[i])

            if sr_input != TARGET_SR:
                input_waveform = torchaudio.transforms.Resample(
                    sr_input, TARGET_SR)(input_waveform)

            if sr_target != TARGET_SR:
                target_waveform = torchaudio.transforms.Resample(
                    sr_target, TARGET_SR)(target_waveform)

            # 입력 및 대상 오디오를 NumPy 배열에 저장 (2채널 오디오)
            data_array[i, 0, :] = input_waveform[0]
            data_array[i, 1, :] = target_waveform[0]

        # NumPy 배열을 .npy 파일로 저장
        output_path = os.path.join(output_dir, f'data_{split}.npy')
        np.save(output_path, data_array)


# 경로 설정
input_dirs = ['./data/train_val_test/noisy/-5db',
              './data/train_val_test/noisy/-0db']
target_dir = './data/train_val_test/clean'
output_dir = './data/npy'

# 출력 디렉토리 생성 (없으면 생성)
os.makedirs(output_dir, exist_ok=True)

# 데이터 저장 함수 호출
save_data_as_npy(input_dirs, target_dir, output_dir)
