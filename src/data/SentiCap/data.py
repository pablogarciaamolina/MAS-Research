import os
import shutil
import pandas as pd
import opendatasets as od

import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from models.text import BertEmbeddings

DATA_PATH = "data/SentiCap"

class SentiCap_Dataset(Dataset):
    '''Customized dataset for the SentiCap dataset.
    It will contain image, text, and sentiment labels.

    Parameters
    ----------
    data_path : str
        The path to the data folder (as such, the data,
        with the audio, text, and emotion files, should
        already be stored there).
    split : str
        The split (train, val or test) in which each dataset is, 
        as it is a known dataste it is divided in train , 
        test and val.
    '''
    def __init__(self, data_path: str, split: str, images_size: tuple[int, int]=(640, 470)):
        # Path in which the images are stored
        self.image_dir = os.path.join(data_path,"Images")
        # Take the dataframe with the info of the files
        self.info_df = pd.read_csv(data_path + "/senticap.csv")
        # Select only the relevant datat for each split
        self.info_df = self.info_df[self.info_df["split"].str.contains(split)]
        # Redefine the data of column tokens so it is recognized as a list
        self.info_df["tokens"] = self.info_df["tokens"].fillna("[]").apply(lambda x: eval(x))

        # Embedding for the text
        self.embedding = BertEmbeddings()

        # Transformation for images
        h, w = images_size
        self.image_transformations = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((h, w))
        ])


    def __len__(self):
        return len(self.info_df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Get image name
        image_name = self.info_df["filename"][idx]
        # Get text tokens
        # txt_tokens = self.info_df["tokens"][idx]
        # Get text
        text_raw = self.info_df["raw"][idx]
        # Get label
        label = self.info_df["sentiment"][idx]
        # Transform the image to a torch.tensor
        img_tensor = self.image_transformations(Image.open(self.image_dir + "/" + image_name)).type(torch.double)
        # Transform text into embedding
        txt_embedded = self.embedding(text_raw).type(torch.double)

        return img_tensor, txt_embedded, torch.tensor(int(label), dtype=torch.long)


def load_data(batch_size: int = 1, shuffle: bool = False,
              num_workers: int = 0) ->\
                  tuple[DataLoader, DataLoader, DataLoader]:

    '''Loads the data from the SentiCap dataset, creating
    the training, validation, and testing dataloaders.

    Parameters
    ----------
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
    # Download data if data/SentiCap is empty
    data_dir = "data/SentiCap"
    
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    if not os.listdir(data_dir):
        os.chdir('./credentials')
        dataset = "https://www.kaggle.com/datasets/prathamsaraf1389/senticap"
        # Move the data to the designated directory and rename it 
        od.download(dataset)
        os.chdir('../')
        os.replace('credentials/senticap/senticap.csv', 'data/SentiCap/senticap.csv')
        shutil.move('credentials/senticap/senticap_images', 'data/SentiCap/senticap_images')
        os.rename('data/SentiCap/senticap_images', 'data/SentiCap/Images')
        shutil.rmtree('credentials/senticap')

    # Create each Dataset
    train_dataset = SentiCap_Dataset(DATA_PATH, 'train')
    test_dataset = SentiCap_Dataset(DATA_PATH, 'test')
    val_dataset = SentiCap_Dataset(DATA_PATH, 'val')
    # Create each dataloader
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=shuffle, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    load_data()