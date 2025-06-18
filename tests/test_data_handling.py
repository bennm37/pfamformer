from pfamformer.data_handling import load, clean_train
import os
import glob
import torch
from pfamformer.data_handling import LazyEmbeddingDataset
from torch.utils.data import DataLoader

def test_load_and_clean():
    train = load("data/random_split/train")
    assert train.shape[0] == 1086741
    test = load("data/random_split/test")
    assert test.shape[0] == 126171
    dev = load("data/random_split/dev")
    assert dev.shape[0] == 126171
    cleaned = clean_train(train)
    assert cleaned.shape[0] == 1085997

def test_lazy_embedding_dataset():
    files = sorted(glob.glob(os.path.join('data/embeddings/test', '*.npy')))
    label_files = sorted(glob.glob(os.path.join('data/random_split/test', 'data-0000*-of-00010')))
    dataset = LazyEmbeddingDataset(files, label_files)
    assert len(dataset) > 0
    embedding, label_idx = dataset[0]
    assert isinstance(embedding, torch.Tensor)
    assert isinstance(label_idx, int)
    loader = DataLoader(dataset, batch_size=4, shuffle=False)
    batch = next(iter(loader))
    assert batch[0].shape == (4,960)
    assert batch[1].shape[0] == 4

if __name__=="__main__":
    test_lazy_embedding_dataset()