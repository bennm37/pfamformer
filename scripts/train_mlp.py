from pfamformer.data_handling import EmbeddingDataset, clean_train, load, get_weighted_sampler
from pfamformer.mlp import MLPClassifier
from torch.utils.data import DataLoader
import glob, os

def create_pfam():
    print(f"Loading train ...")
    train_df = clean_train(load(f"data/random_split/train"))
    train_set = EmbeddingDataset(train_df, f"data/embeddings/train")
    print(f"Loading dev ...")
    dev_df = load(f"data/random_split/dev")
    dev_set = EmbeddingDataset(dev_df, f"data/embeddings/dev")
    print(f"Loading test ...")
    test_df = load(f"data/random_split/test", f"data/embeddings/test")
    test_set = EmbeddingDataset(test_df)
    return train_set, dev_set, test_set

train_set, dev_set, test_set = create_pfam("train")
train_batch = DataLoader(train_set, batch_size=128, shuffle=True)
dev_batch = DataLoader(train_set, batch_size=128, shuffle=True)
mlp = MLPClassifier(960, 17929)
mlp.train_model(train_batch, epochs=30, lr=1e-3)
