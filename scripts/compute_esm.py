from pfamformer.data_handling import load, clean_train
import numpy as np
import os
import pandas as pd
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig

CLIENT = ESMC.from_pretrained("esmc_300m").to("cpu") # need cuda on nemo gpus

def compute_embeddings(df):
    n_rows = df.shape[0]
    embeddings = np.zeros((n_rows, 960, 2))
    for i, sequence in enumerate(df["sequence"].values[:20]):
        if i%10==0:
            print(f"Starting Embedding {i}")
        protein = ESMProtein(sequence=sequence)
        protein_tensor = CLIENT.encode(protein)
        logits_output = CLIENT.logits(
        protein_tensor, LogitsConfig(sequence=True, return_embeddings=True)
        )
        embedding = logits_output.embeddings.numpy()
        embeddings[i, :, 0] = np.mean(embedding[0], axis=0)
        embeddings[i, :, 1] = np.max(embedding[0], axis=0)
    return embeddings

def compute_embeddings_folder(folder_name, results_folder):
    filenames = sorted(os.listdir(folder_name))
    for filename in filenames:
        print(f"Starting file {filename}")
        df = pd.read_csv(f"{folder_name}/{filename}")
        embeddings = compute_embeddings(df)
        np.save(f"{results_folder}/{filename}_embeddings", embeddings)

compute_embeddings_folder("./data/random_split/train", "./data/embeddings/train")
compute_embeddings_folder("./data/random_split/test", "./data/embeddings/test")
compute_embeddings_folder("./data/random_split/dev", "./data/embeddings/dev")


