import pandas as pd
import numpy as np
import os
import torch
from torch.utils.data import Dataset
import pandas as pd
import glob
from tqdm import tqdm

def load(folder_name):
    """Load the chunked data."""
    chunk_names = sorted(os.listdir(folder_name))
    df = pd.concat([pd.read_csv(f"{folder_name}/{name}") for name in chunk_names])
    df["accession_no"] = [val.split(".")[0][2:] for val in df["family_accession"].values]
    df.index = range(len(df.index))
    return df

def clean_train(df):
    """Drop duplicates sequences and remove duplicated sequences with different labels."""
    dup_sequences = df.duplicated(subset=['sequence'],keep=False)
    dup_sequences_and_labels = df.duplicated(subset=['sequence','accession_no'], keep=False)
    df.drop(df[dup_sequences & ~dup_sequences_and_labels].index, inplace=True)
    df.drop_duplicates(subset=['sequence'], inplace=True)
    return df

def get_lengths(file_list):
        lengths = []
        for f in file_list:
             arr = np.load(f, mmap_mode='r')
             lengths.append(arr.shape[0])
        return lengths

def get_embeddings(index, pattern=f'data/embeddings/train/*.npy'):
    file_list = sorted(glob.glob(pattern))
    lengths = get_lengths(file_list)
    cum_lengths = np.cumsum([0]+lengths)
    prev_file_index = -1
    embeddings = np.zeros((len(index), 960, 2))
    print("Loading Embeddings ... ")
    for i, ind in enumerate(tqdm(index)):
        file_index = np.searchsorted(cum_lengths, ind, side='right') - 1
        local_index = ind - cum_lengths[file_index]
        if file_index > prev_file_index:
            c_embeddings = np.load(file_list[file_index])
        embeddings[i] = c_embeddings[local_index].astype(np.float32)
        prev_file_index = file_index
    return embeddings


class LazyEmbeddingDataset(Dataset):
    def __init__(self, file_list, label_files, pooling="max"):
        self.file_list = file_list
        self.lengths = []
        self.lengths = get_lengths(self.file_list)
        self.cum_lengths = np.cumsum([0] + self.lengths)
        dfs = [pd.read_csv(f, header=None) for f in label_files]
        labels = pd.concat(dfs, ignore_index=True)[0].tolist()
        try:
            self.pooling = ["mean","max"].index(pooling)
        except ValueError:
            raise ValueError(f"Unknown value for pooling: {self.pooling}")
        self.accessions = [val.split(".")[0][2:] for val in labels]
        self.unique_labels = sorted(set(self.accessions))
        self.label_to_index = {l: i for i, l in enumerate(self.unique_labels)}
        self.label_indices = [self.label_to_index[a] for a in self.accessions]

    def __len__(self):
        return self.cum_lengths[-1]

    def __getitem__(self, index):
        file_index = np.searchsorted(self.cum_lengths, index, side='right') - 1
        local_index = index - self.cum_lengths[file_index]
        arr = np.load(self.file_list[file_index], mmap_mode='r')
        embedding = torch.from_numpy(arr[local_index].astype(np.float32)[:,self.pooling])
        # from profiling, 
        label_index = self.label_indices[index]
        return embedding, label_index
    

class EmbeddingDataset(Dataset):
    def __init__(self, df, pooling="mean"):
        self.pooling_type = ["mean","max"].index(pooling)
        self.embeddings = torch.from_numpy(get_embeddings(df.index)[:,:, self.pooling_type].astype(np.float32))
        self.length = self.embeddings.shape[0]
        self.accession_nos = df["accession_no"]
        self.unique_labels = sorted(set(self.accession_nos))
        self.label_to_index = {l: i for i, l in enumerate(self.unique_labels)}
        self.label_indices = torch.tensor([self.label_to_index[a] for a in self.accession_nos])

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        embedding = self.embeddings[index]
        label_index = self.label_indices[index]
        return embedding, label_index