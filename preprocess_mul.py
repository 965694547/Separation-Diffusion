import os
import librosa
import pyworld as pw
import numpy as np
import speechbrain as sb
import sys
import torch
from multiprocessing.pool import Pool
from tqdm import tqdm

from hyperpyyaml import load_hyperpyyaml

version = 'wav8k/min/'
set_types = ["train-360", "dev", "test"]
s = ['s1', 's2', 's3']
thread_num = os.cpu_count()

def f0_to_coarse(f0):
    f0_mel = 1127 * np.log(1 + f0 / 700)
    f0_mel_min = 1127 * np.log(1 + hparams['f0_min'] / 700)
    f0_mel_max = 1127 * np.log(1 + hparams['f0_max'] / 700)
    # f0_mel[f0_mel == 0] = 0
    # 大于0的分为255个箱
    f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * (hparams['f0_bin'] - 2) / (f0_mel_max - f0_mel_min) + 1

    f0_mel[f0_mel < 0] = 1
    f0_mel[f0_mel > hparams['f0_bin'] - 1] = hparams['f0_bin'] - 1
    f0_coarse = np.rint(f0_mel).astype(np.int)
    # print('Max f0', np.max(f0_coarse), ' ||Min f0', np.min(f0_coarse))
    assert (np.max(f0_coarse) <= 256 and np.min(f0_coarse) >= 0)
    return f0_coarse


def get_pitch(wav_data, mel, hparams):
    """

    :param wav_data: [T]
    :param mel: [T, 80]
    :param hparams:
    :return:
    """
    _f0, t = pw.dio(wav_data.astype(np.double), hparams['sample_rate'],
                    frame_period=hparams['hop_size'] / hparams['sample_rate'] * 1000)
    f0 = pw.stonemask(wav_data.astype(np.double), _f0, t, hparams['sample_rate'])  # pitch refinement
    delta_l = len(mel) - len(f0)
    assert np.abs(delta_l) <= 2
    if delta_l > 0:
        f0 = np.concatenate([f0] + [f0[-1]] * delta_l)
    f0 = f0[:len(mel)]
    pitch_coarse = f0_to_coarse(f0) + 1
    return f0, pitch_coarse


def process_f0(f0, hparams):
    f0_ = (f0 - hparams['f0_mean']) / hparams['f0_std']
    f0_[f0 == 0] = np.interp(np.where(f0 == 0)[0], np.where(f0 > 0)[0], f0_[f0 > 0])
    uv = (torch.FloatTensor(f0) == 0).float()
    f0 = f0_
    f0 = torch.FloatTensor(f0)
    return f0, uv


def process_mel_pitch_f0(wav, hparams):
    # get amplitude spectrogram
    x_stft = librosa.stft(wav, n_fft=hparams['fft_size'], hop_length=hparams['hop_size'],
                          win_length=hparams['win_length'], window=hparams['window'], pad_mode="constant")
    spc = np.abs(x_stft)  # (n_bins, T)

    # get mel basis
    fmin = 0 if hparams['fmin'] == -1 else hparams['fmin']
    fmax = hparams['sample_rate'] / 2 if hparams['fmax'] == -1 else hparams['fmax']
    mel_basis = librosa.filters.mel(sr=hparams['sample_rate'], n_fft=hparams['fft_size'], n_mels=hparams['num_mels'], fmin=fmin, fmax=fmax)
    mel = mel_basis @ spc
    mel = mel.T

    f0, pitch_coarse = get_pitch(wav, mel, hparams)
    f0 = f0[:len(mel)]
    pitch = f0_to_coarse(f0) + 1

    return mel, pitch, f0

def chunk_process(wave_list, in_dir, hparams, mel_dir, pitch_dir, f0_dir):
    f0s_ = []
    for wav_name in tqdm(wave_list):
        if ".wav" not in wav_name:
            continue

        basename = wav_name.split(".")[0]
        if isinstance(wav_name, str):
            wav, _ = librosa.core.load(os.path.join(in_dir, wav_name), sr=hparams['sample_rate'])
        else:
            wav = wav_name

        mel, pitch, f0 = process_mel_pitch_f0(wav, hparams)
        f0s_.append(f0)

        np.save(f'{mel_dir}/{basename}.npy', mel)
        np.save(f'{pitch_dir}/{basename}.npy', pitch)
        np.save(f'{f0_dir}/{basename}.npy', f0)
    return f0s_

if __name__ == "__main__":
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    if hparams['data_folder'] == 'Libri2Mix' or hparams['data_folder'] == 'Libri3Mix':
        f0s = []
        f0s_dir = os.path.join(hparams['data_folder'], version)
        for set in set_types:
            out_dir = os.path.join(hparams['data_folder'], version, set)
            for i in range(hparams['num_spks']):
                in_dir = os.path.join(out_dir, s[i])
                mel_dir = os.path.join(in_dir, "mel")
                pitch_dir = os.path.join(in_dir, "pitch")
                #energy_dir = os.path.join(out_dir, "energy")
                f0_dir = os.path.join(in_dir, "f0")
                #uv_dir = os.path.join(out_dir, "uv")
                os.makedirs(mel_dir, exist_ok=True)
                os.makedirs(pitch_dir, exist_ok=True)
                #os.makedirs(energy_dir, exist_ok=True)
                os.makedirs(f0_dir, exist_ok=True)
                #os.makedirs(uv_dir, exist_ok=True)

                p = Pool(thread_num)
                thread_list = list()
                for wave_list in np.array_split(os.listdir(in_dir), thread_num):
                    t = p.apply_async(chunk_process, args=[wave_list, in_dir, hparams, mel_dir, pitch_dir, f0_dir])
                    thread_list.append(t)
                p.close()
                for t in thread_list:
                    f0s_ = t.get()
                    f0s.extend(f0s_)
                thread_list = list()
                p.join()

        f0s = np.concatenate(f0s, 0)
        f0s = f0s[f0s != 0]
        np.save(f'{f0s_dir}/f0s_mean.npy', np.mean(f0s).item())
        np.save(f'{f0s_dir}/f0s_std.npy', np.std(f0s).item())




