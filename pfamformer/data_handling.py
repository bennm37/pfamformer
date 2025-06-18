import pandas as pd
import numpy as np
import os
import torch
from torch.utils.data import Dataset
import pandas as pd

def load(folder_name):
    """Load the chunked data."""
    chunk_names = os.listdir(folder_name)
    df = pd.concat([pd.read_csv(f"{folder_name}/{name}") for name in chunk_names])
    df["accession_no"] = [val.split(".")[0][2:] for val in df["family_accession"].values]
    return df

def clean_train(df):
    """Drop duplicates sequences and remove duplicated sequences with different labels."""
    dup_sequences = df.duplicated(subset=['sequence'],keep=False)
    dup_sequences_and_labels = df.duplicated(subset=['sequence','accession_no'], keep=False)
    df.drop(df[dup_sequences & ~dup_sequences_and_labels].index, inplace=True)
    df.drop_duplicates(subset=['sequence'], inplace=True)
    return df


class LazyEmbeddingDataset(Dataset):
    def __init__(self, file_list, label_files, pooling="max"):
        self.file_list = file_list
        self.lengths = []
        for f in self.file_list:
            arr = np.load(f, mmap_mode='r')
            self.lengths.append(arr.shape[0])
        self.cum_lengths = np.cumsum([0] + self.lengths)
        dfs = [pd.read_csv(f, header=None) for f in label_files]
        labels = pd.concat(dfs, ignore_index=True)[0].tolist()
        try:
            self.pooling = ["mean","max"].index(pooling)
        except ValueError:
            raise ValueError(f"Unknown value for pooling: {self.pooling}")
        self.accessions = [val.split(".")[0][2:] for val in labels]
        self.unique_labels = sorted(set(self.accessions))
        self.label_to_idx = {l: i for i, l in enumerate(self.unique_labels)}
        self.label_indices = [self.label_to_idx[a] for a in self.accessions]

    def __len__(self):
        return self.cum_lengths[-1]

    def __getitem__(self, idx):
        file_idx = np.searchsorted(self.cum_lengths, idx, side='right') - 1
        local_idx = idx - self.cum_lengths[file_idx]
        arr = np.load(self.file_list[file_idx], mmap_mode='r')
        embedding = torch.from_numpy(arr[local_idx].copy()[:,self.pooling])
        label_idx = self.label_indices[idx]
        return embedding, label_idx