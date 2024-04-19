import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
import os
import torch.nn.functional as F

# Audio processing
from scipy import signal
from scipy.io import wavfile

from .file_management import processed_tensors_management
from models.text.bertEmbeddings import BertEmbeddings


DATA_PATH = "data/IEMOCAP"
TEST_SIZE = 0.2
VAL_SIZE = 0.2


class IEMOCAP_Dataset(Dataset):
    '''Customized dataset for the IEMOCAP dataset.
    It will contain audio, text, and emotion labels.
    Additionally, it will apply preprocessing to both
    audio (by converting it to a spectrogram) and text
    (by tokenizing it using the BERT tokenizer).

    Parameters
    ----------
    data_path : str
        The path to the data folder (as such, the data,
        with the audio, text, and emotion files, should
        already be stored there).
    audio_time : int
        The length of the audio files (in periods of sampling) to
        consider. Default is 1520, which corresponds to 1520, which
        is the max length of the audio files in our dataset.
    '''

    def __init__(self, data_path: str, audio_time: int = 1520):
        
        self.audio_time = audio_time
        # Paths to the folders
        self.audio_dir = os.path.join(data_path, "Audio")
        self.text_dir = os.path.join(data_path, "Text")
        self.emotion_dir = os.path.join(data_path, "Emotion/Utterances")

        # Get the list of files
        self.files = os.listdir(self.audio_dir)
        # Take only the file name (without the extension)
        self.files = [file.split(".")[0] for file in self.files]
        
        self.bert = BertEmbeddings()

        # Process data
        self._save_embeddings_and_spectrograms()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Get the file name
        file = self.files[idx]

        path = DATA_PATH + "/" + "Processed_tensors" + "/" + file

        text: torch.Tensor = torch.load(path + "/" + "text.pt")
        audio: torch.Tensor = torch.load(path + "/" + "audio.pt")
        emotion: torch.Tensor = torch.load(path + "/" + "emotion.pt")
    
        return  audio, text, emotion
    
    def _save_embeddings_and_spectrograms(self) -> None:
        """
        Saves in memory the processed data, both the BERT embedded text inputs and
        the spectrograms of the audio.
        """

        processed_tensors_management(self.files)

        for file in self.files:
            path: str = DATA_PATH + "/" + "Processed_tensors" + "/" + file
            # Text
            text_file = os.path.join(self.text_dir, file + ".txt")
            with open(text_file, "r") as f:
                text = f.read()
            text_tensor = self.bert(text).type(torch.double)
            torch.save(text_tensor, path + "/" + "text.pt")

            # Audio
            audio_file = os.path.join(self.audio_dir, file + ".wav")
            # Transform the audio file to a spectrogram
            spectrogram = audio2spectrogram(audio_file)
            spectrogram = get_3d_spec(spectrogram)
            # Transpose to match PyTorch's format (F, T, C)
            npimg = np.transpose(spectrogram, (1, 0, 2))
            # Convert to tensor
            audio: torch.Tensor = torch.tensor(npimg)
            # Pad and cut to maximum length
            if audio.size(1) < self.audio_time:
                pad_amount = self.audio_time - audio.size(1)
                # Fill with zeros along the time axis
                audio = F.pad(audio, (0, 0, 0, pad_amount), "constant", 0).type(torch.double)
            else:
                audio = audio[:, :self.audio_time, :].type(torch.double)
            torch.save(audio, path + "/" + "audio.pt")

            # Emotion
            emotion_file = os.path.join(self.emotion_dir, file + ".txt")
            with open(emotion_file, "r") as f:
                emotion = f.read()
            # Convert the emotion to an integer (0-9)
            emotion = torch.tensor(int(emotion), dtype=torch.long)
            torch.save(emotion, path + "/" + "emotion.pt")



def log_specgram(audio: np.array, sample_rate: int, window_size: int = 20,
                 step_size: int = 10, eps: float = 1e-10):
    '''Computes the log of the spectrogram of audio data.
    This will be used to convert the audio (.wav) files into
    spectrograms (which are much easier to work with).

    Parameters
    ----------
    audio : np.array
        The audio data
    sample_rate : int
        The sample rate of the audio data (in Hz)
    window_size : int
        The size of the window for the FFT (in ms)
    step_size : int
        The size of the step between windows (in ms)
    eps : float
        A small value to avoid log(0)

    Returns
    -------
    freqs : np.array
        The frequencies of the spectrogram
    spectrogram : np.array
        The log of the spectrogram
    '''
    nperseg: int = int(round(window_size * sample_rate / 1e3))
    noverlap: int = int(round(step_size * sample_rate / 1e3))
    freqs, _, spec = signal.spectrogram(audio,
                                        fs=sample_rate,
                                        window='hann',
                                        nperseg=nperseg,
                                        noverlap=noverlap,
                                        detrend=False)
    return freqs, np.log(spec.T.astype(np.float32) + eps)


def audio2spectrogram(filepath: str):
    '''Converts an audio file to a spectrogram.

    Parameters
    ----------
    filepath : str
        The path to the audio file

    Returns
    -------
    spectrogram : np.array
        The log of the spectrogram
    '''
    # Read the audio file (memory-mapped to avoid loading the whole file)
    samplerate, test_sound = wavfile.read(filepath, mmap=True)
    # Compute the spectrogram
    _, spectrogram = log_specgram(test_sound, samplerate)
    return spectrogram


def get_3d_spec(Sxx_in, moments=None):
    '''Converts a spectrogram to a 3D tensor, with dimensions
    (height, width, channels). The channels are the base spectrogram,
    the first derivative, and the second derivative.

    Parameters
    ----------
    Sxx_in : np.array
        The input spectrogram
    moments : tuple
        The mean and standard deviation of the base spectrogram, the first
        derivative, and the second derivative. If None, the mean is 0 and
        the standard deviation is 1.

    Returns
    -------
    np.array
        The 3D tensor
    '''
    # Compute the first and second derivatives
    if moments is not None:
        (base_mean, base_std, delta_mean, delta_std,
            delta2_mean, delta2_std) = moments
    # If moments is not provided, assume mean=0 and std=1
    else:
        base_mean, delta_mean, delta2_mean = (0, 0, 0)
        base_std, delta_std, delta2_std = (1, 1, 1)
    h, w = Sxx_in.shape

    right1 = np.concatenate([Sxx_in[:, 0].reshape((h, -1)),
                             Sxx_in], axis=1)[:, :-1]
    delta = (Sxx_in - right1)[:, 1:]
    delta_pad = delta[:, 0].reshape((h, -1))
    delta = np.concatenate([delta_pad, delta], axis=1)
    right2 = np.concatenate([delta[:, 0].reshape((h, -1)),
                             delta], axis=1)[:, :-1]
    delta2 = (delta - right2)[:, 1:]
    delta2_pad = delta2[:, 0].reshape((h, -1))
    delta2 = np.concatenate([delta2_pad, delta2], axis=1)
    base = (Sxx_in - base_mean) / base_std
    delta = (delta - delta_mean) / delta_std
    delta2 = (delta2 - delta2_mean) / delta2_std
    stacked: list = [arr.reshape((h, w, 1)) for
                     arr in (base, delta, delta2)]
    return np.concatenate(stacked, axis=2)


def load_data(time_dim: int = 1500, batch_size: int = 1, shuffle: bool = False,
              num_workers: int = 0) ->\
        tuple[DataLoader, DataLoader, DataLoader]:
    
    '''Loads the data from the IEMOCAP dataset, creating
    the training, validation, and testing dataloaders.

    Parameters
    ----------
    time_dim:
        Size for audio time dimension
    batch_size : int
        The batch size for the DataLoader
    shuffle : bool
        Whether to shuffle the training data
    num_workers : int
        The number of workers for the DataLoader

    Returns
    -------
    train_loader : DataLoader
        The training dataloader
    val_loader : DataLoader
        The validation dataloader
    test_loader : DataLoader
        The testing dataloader
    '''
    # Create the dataset
    dataset = IEMOCAP_Dataset(DATA_PATH, audio_time=time_dim)

    # Split the dataset into training and testing
    test_size = int(TEST_SIZE * len(dataset))
    train_dataset, test_dataset = random_split(dataset,
                                               [len(dataset) - test_size,
                                                test_size])

    # Divide the training dataset into training and validation
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset,
                                              [train_size, val_size])

    # Create the dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=shuffle, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader