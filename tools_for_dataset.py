import os
import soundfile
import numpy as np


def scan_directory(dir_name):
    if os.path.isdir(dir_name) is False:
        print("[Error] There is no directory '%s'." % dir_name)
        exit()

    addrs = []
    for subdir, dirs, files in os.walk(dir_name):
        for file in files:
            if file.endswith(".wav"):
                filepath = subdir + file
                addrs.append(filepath)
    return

def find_pair(noisy_dirs):
    clean_dirs = []
    for i in range(len(noisy_dirs)):
        addrs = noisy_dirs[i]
        if addrs.endswith(".wav"):
            addr_noisy = str(addrs)
            addr_clean = str(addrs).replace('noisy', 'clean')
            clean_dirs.append(addr_clean)
    return clean_dirs

def addr2wav(addr):
    wav, fs = soundfile.read(addr)
    # normalize
    wav = minMaxNorm(wav)
    return
    #######################################################################

#                        Data Normalization                           #
#######################################################################
def minMaxNorm(wav, eps=1e-8):
    max = np.max(abs(wav))
    min = np.min(abs(wav))
    wav = (wav - min) / (max - min + eps)
    return