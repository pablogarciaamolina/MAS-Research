import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os

# Audio processing
from scipy import signal
from scipy.io import wavfile
import matplotlib.pyplot as plt

def log_specgram(audio, sample_rate, window_size=40,
                 step_size=20, eps=1e-10):
    nperseg = int(round(window_size * sample_rate / 1e3))
    noverlap = int(round(step_size * sample_rate / 1e3))
    freqs, _, spec = signal.spectrogram(audio,
                                    fs=sample_rate,
                                    window='hann',
                                    nperseg=nperseg,
                                    noverlap=noverlap,
                                    detrend=False)
    return freqs, np.log(spec.T.astype(np.float32) + eps)


def audio2spectrogram(filepath):
    samplerate, test_sound  = wavfile.read(filepath,mmap=True)
    _, spectrogram = log_specgram(test_sound, samplerate)
    return spectrogram


def get_3d_spec(Sxx_in, moments=None):
    if moments is not None:
        (base_mean, base_std, delta_mean, delta_std,
             delta2_mean, delta2_std) = moments
    else:
        base_mean, delta_mean, delta2_mean = (0, 0, 0)
        base_std, delta_std, delta2_std = (1, 1, 1)
    h, w = Sxx_in.shape
    right1 = np.concatenate([Sxx_in[:, 0].reshape((h, -1)), Sxx_in], axis=1)[:, :-1]
    delta = (Sxx_in - right1)[:, 1:]
    delta_pad = delta[:, 0].reshape((h, -1))
    delta = np.concatenate([delta_pad, delta], axis=1)
    right2 = np.concatenate([delta[:, 0].reshape((h, -1)), delta], axis=1)[:, :-1]
    delta2 = (delta - right2)[:, 1:]
    delta2_pad = delta2[:, 0].reshape((h, -1))
    delta2 = np.concatenate([delta2_pad, delta2], axis=1)
    base = (Sxx_in - base_mean) / base_std
    delta = (delta - delta_mean) / delta_std
    delta2 = (delta2 - delta2_mean) / delta2_std
    stacked = [arr.reshape((h, w, 1)) for arr in (base, delta, delta2)]
    return np.concatenate(stacked, axis=2)


class IEMOCAP_Dataset(Dataset):
    def __init__(self, data_path: str):
        self.audio_dir = os.path.join(data_path, "Audio")
        self.text_dir = os.path.join(data_path, "Text")
        self.emotion_dir = os.path.join(data_path, "Emotion/Utterances")

        self.audio_files = os.listdir(self.audio_dir)
        self.text_files = os.listdir(self.text_dir)
        self.emotion_files = os.listdir(self.emotion_dir)

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        # Text
        text_file = os.path.join(self.text_dir, self.text_files[idx])
        with open(text_file, "r") as f:
            text = f.read()

        # Audio
        audio_file = os.path.join(self.audio_dir, self.audio_files[idx])
        spectrogram = audio2spectrogram(audio_file)
        spectrogram = get_3d_spec(spectrogram)
        # Transpose to match PyTorch's format
        npimg = np.transpose(spectrogram, (2, 0, 1))
        audio: torch.Tensor = torch.tensor(npimg)

        # Emotion
        emotion_file = os.path.join(self.emotion_dir, self.emotion_files[idx])
        with open(emotion_file, "r") as f:
            emotion = f.read()
        emotion = int(emotion)

        return audio, text, emotion


if __name__ == "__main__":
    path = "data/"
    dataset = IEMOCAP_Dataset(path)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
