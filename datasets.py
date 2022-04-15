from torchvision import transforms
import torch.utils.data as data
from PIL import Image
import pandas as pd
import numpy as np
import torch
import cv2
import os


to_tensor = transforms.ToTensor()


class TAN_DATASET(data.Dataset):

    def __init__(self, root, indices, df_path, batch_size, transforms = None):

        self.root = root
        self.indices = indices
        self.df = pd.read_csv(df_path)
        self.batch_size = batch_size
        self.im_size = (64, 64)
        self.data = self.load_samples()
        
    def __getitem__(self, index):
        batch = self.data[index]        
        batch_imagery = Image.open(batch[0]).convert("RGB")
        batch_imagery, batch_labels = to_tensor(np.array(batch_imagery.resize(self.im_size, Image.ANTIALIAS))), torch.tensor(batch[1]).unsqueeze(0)
        return batch_imagery, batch_labels

    def __len__(self):
        return len(self.data)

    def load_samples(self):
        
        files = [_ for _ in os.listdir(self.root) if _.endswith(".png")]
        files = [files[_] for _ in self.indices]
        school_ids = [i.split("_")[1].strip(".png") for i in files]
        files = [self.root + i for i in files if ".ipynb" not in i]
                
        df = self.df[self.df["school_id"].isin(school_ids)]
        df = df[["school_id", "binary"]].set_index("school_id").reindex(school_ids)
        labels = df["binary"].to_list()
        data = list(zip(files, labels))
        
        return data
    
    

class MEX_DATASET(data.Dataset):

    def __init__(self, root, indices, df_path, batch_size, transforms = None):
        
        self.root = root
        self.indices = indices
        self.df = pd.read_csv(df_path)
        self.batch_size = batch_size
        self.im_size = (64, 64)
        self.data = self.load_samples()
        
    def __getitem__(self, index):
        batch = self.data[index]        
        batch_imagery = Image.open(batch[0]).convert("RGB")
        batch_imagery, batch_labels = to_tensor(np.array(batch_imagery.resize(self.im_size, Image.ANTIALIAS))), torch.tensor(batch[1]).unsqueeze(0)
        return batch_imagery, batch_labels

    def __len__(self):
        return len(self.data)

    def load_samples(self):
        
        files = [_ for _ in os.listdir(self.root) if _.endswith(".png")]
        files = [files[_] for _ in self.indices]
        school_ids = [i.split("_")[1].strip(".png") for i in files]
        files = [self.root + i for i in files if ".ipynb" not in i]
                
        df = self.df[self.df["school_id"].isin(school_ids)]
        df = df[["school_id", "binary"]].set_index("school_id").reindex(school_ids)
        labels = df["binary"].to_list()
        data = list(zip(files, labels))
        
        return data