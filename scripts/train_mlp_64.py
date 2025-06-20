from pfamformer.data_handling import LazyEmbeddingDataset, EmbeddingDataset, load, clean_train, get_lengths
from pfamformer.mlp import MLPClassifier
from torch.utils.data import DataLoader
import glob, os
import numpy as np

def create_set(set_type):
    embeddings = sorted(glob.glob(f'data/embeddings/{set_type}/*.npy'))
    label_names = [os.path.basename(e).split("_")[0] for e in embeddings]
    labels = [os.path.join(f"data/random_split/{set_type}/{name}") for name in label_names]
    return LazyEmbeddingDataset(embeddings, labels)

def create_pfam_64_small():
    print(f"Loading train ...")
    train_df = clean_train(load(f"data/random_split/train"))
    grouped = train_df.groupby(by="accession_no").size()
    nos, _ = zip(*sorted(grouped.items(), key=lambda x: x[1], reverse=True))
    top_64 = nos[:64]
    train_df_64 = train_df[train_df["accession_no"].isin(top_64)]
    train_set = EmbeddingDataset(train_df_64)
    print(f"Loading dev ...")
    dev_df_64 = load(f"data/random_split/dev")
    dev_df_64 = dev_df_64[dev_df_64["accession_no"].isin(top_64)]
    dev_set = EmbeddingDataset(dev_df_64)
    print(f"Loading test ...")
    test_df_64 = load(f"data/random_split/test")
    test_df_64 = test_df_64[test_df_64["accession_no"].isin(top_64)]
    test_set = EmbeddingDataset(test_df_64)
    return train_set, dev_set, test_set

if __name__=="__main__":
    mlp = MLPClassifier(960, 64)
    train_set, dev_set, test_set = create_pfam_64_small()
    train_batch = DataLoader(train_set, batch_size=128, shuffle=True)
    mlp.train_model(train_batch, epochs=30, lr=1e-3)