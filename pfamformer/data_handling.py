import pandas as pd
import numpy as np
import os
import torch
from torch.utils.data import Dataset, WeightedRandomSampler
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

def get_weighted_sampler(df):
    class_counts = df['accession_no'].value_counts()
    class_weights = 1.0 / class_counts
    sample_weights = df['accession_no'].map(class_weights).values
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    return sampler



class EmbeddingDataset(Dataset):
    def __init__(self, df, embedding_folder, pooling="mean", label_to_index=None):
        self.pooling_type = ["mean","max"].index(pooling)
        pattern = f"{embedding_folder}/*.npy"
        self.embeddings = torch.from_numpy(get_embeddings(df.index, pattern)[:,:, self.pooling_type].astype(np.float32))
        self.length = self.embeddings.shape[0]
        self.accession_nos = df["accession_no"]
        if label_to_index is None:
            self.unique_labels = sorted(set(self.accession_nos))
            self.label_to_index = {l: i for i, l in enumerate(self.unique_labels)}
        else:
            self.label_to_index = label_to_index
        self.label_indices = torch.tensor([self.label_to_index[a] for a in self.accession_nos])

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        embedding = self.embeddings[index]
        label_index = self.label_indices[index]
        return embedding, label_index