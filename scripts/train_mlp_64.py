from pfamformer.data_handling import LazyEmbeddingDataset, EmbeddingDataset, load, clean_train, get_lengths
from pfamformer.mlp import MLPClassifier
from torch.utils.data import DataLoader
import glob, os
import numpy as np
import pickle

def create_set(set_type):
    embeddings = sorted(glob.glob(f'data/embeddings/{set_type}/*.npy'))
    label_names = [os.path.basename(e).split("_")[0] for e in embeddings]
    labels = [os.path.join(f"data/random_split/{set_type}/{name}") for name in label_names]
    return LazyEmbeddingDataset(embeddings, labels)

def create_pfam_subset(n=64, data="data"):
    print(f"Loading train ...")
    train_df = clean_train(load(f"{data}/random_split/train"))
    grouped = train_df.groupby(by="accession_no").size()
    nos, _ = zip(*sorted(grouped.items(), key=lambda x: x[1], reverse=True))
    top_n = nos[:n]
    train_df_n = train_df[train_df["accession_no"].isin(top_n)]
    train_set = EmbeddingDataset(train_df_n, f"{data}/embeddings/train")
    label_to_index = train_set.label_to_index
    print(f"Loading dev ...")
    dev_df_n = load(f"{data}/random_split/dev")
    dev_df_n = dev_df_n[dev_df_n["accession_no"].isin(top_n)]
    dev_set = EmbeddingDataset(dev_df_n, f"{data}/embeddings/dev", label_to_index=label_to_index)
    print(f"Loading test ...")
    test_df_n = load(f"{data}/random_split/test")
    test_df_n = test_df_n[test_df_n["accession_no"].isin(top_n)]
    test_set = EmbeddingDataset(test_df_n, f"{data}/embeddings/test", label_to_index=label_to_index)
    return train_set, dev_set, test_set

def evaluate_trained(model_path):
    train_df = clean_train(load(f"data/random_split/train"))
    grouped = train_df.groupby(by="accession_no").size()
    nos, _ = zip(*sorted(grouped.items(), key=lambda x: x[1], reverse=True))
    top_64 = nos[:64]
    test_df_64 = load(f"data/random_split/test")
    test_df_64 = test_df_64[test_df_64["accession_no"].isin(top_64)]
    test_set = EmbeddingDataset(test_df_64, f"data/embeddings/test")
    test_batch = DataLoader(test_set, batch_size=128)
    mlp = pickle.load(open(model_path, "rb"))
    df = mlp.evaluate_test_set(test_batch)
    df["epoch"] = 30 
    mlp.log(df, 30)

if __name__=="__main__":
    n = 100
    mlp = MLPClassifier(960, n)
    train_set, dev_set, test_set = create_pfam_subset(200)
    train_batch = DataLoader(train_set, batch_size=128, shuffle=True)
    dev_batch = DataLoader(dev_set, batch_size=128, shuffle=True)
    mlp.train_model(train_batch, dev_dataloader=dev_batch, epochs=10, lr=1e-3)
    # evaluate_trained("data/trained/mlp_2025-06-20_12-15-15.pkl")