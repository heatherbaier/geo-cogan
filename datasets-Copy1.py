from torchvision import transforms
import torch.utils.data as data
import pandas as pd
import torch
import cv2
import os


to_tensor = transforms.ToTensor()


class TAN_DATASET(data.Dataset):

    def __init__(self, root, df_path, batch_size, transforms = None):

        self.root = root
        self.df = pd.read_csv(df_path)
        self.batch_size = batch_size
        self.data = self.load_samples()
        
    def __getitem__(self, index):
        batch = self.data[index]
        batch_imagery, batch_labels = [i[0] for i in batch], [torch.tensor(i[1]).unsqueeze(0) for i in batch]
        batch_imagery = [to_tensor(cv2.imread(i)).unsqueeze(0) for i in batch_imagery]
        batch_imagery, batch_labels = torch.cat(batch_imagery), torch.cat(batch_labels).view(-1, 1)
        return batch_imagery, batch_labels

    def __len__(self):
        return len(self.data)

    def load_samples(self):
        
        files = os.listdir(self.root)
        files = [i for i in files if ".ipynb" not in i]
        school_ids = [i.split("_")[1].strip(".png") for i in files]
        files = [self.root + i for i in files if ".ipynb" not in i]
                
        df = self.df[self.df["YEAR_OF_RESULT"] == 2014]
        df = df[df["CODE"].isin(school_ids)]
        df = df[["CODE", "AVG_MARK"]].set_index("CODE").reindex(school_ids)

        labels = df["AVG_MARK"].to_list()
        
        data = list(zip(files, labels))
        data = [data[i:i + self.batch_size] for i in range(0, len(data), self.batch_size)] 

        return data
    
    

class MEX_DATASET(data.Dataset):

    def __init__(self, root, df_path, batch_size, transforms = None):

        self.root = root
        self.df = pd.read_csv(df_path)
        self.batch_size = batch_size
        self.data = self.load_samples()
        
    def __getitem__(self, index):
        batch = self.data[index]
        batch_imagery, batch_labels = [i[0] for i in batch], [torch.tensor(i[1]).unsqueeze(0) for i in batch]
        batch_imagery = [to_tensor(cv2.imread(i)).unsqueeze(0) for i in batch_imagery]
        batch_imagery, batch_labels = torch.cat(batch_imagery), torch.cat(batch_labels).view(-1, 1)
        return batch_imagery, batch_labels

    def __len__(self):
        return len(self.data)

    def load_samples(self):
        
        files = os.listdir(self.root)
        files = [i for i in files if ".ipynb" not in i]
        school_ids = [i.split("_")[1].strip(".png") for i in files]
        files = [self.root + i for i in files if ".ipynb" not in i]
                
#         df = self.df[self.df["YEAR_OF_RESULT"] == 2014]
        df = self.df[self.df["SchoolID"].isin(school_ids)]
        df = df[["SchoolID", "total_score"]].set_index("SchoolID").reindex(school_ids)

        labels = df["total_score"].to_list()
        
        data = list(zip(files, labels))
        data = [data[i:i + self.batch_size] for i in range(0, len(data), self.batch_size)] 

        return data