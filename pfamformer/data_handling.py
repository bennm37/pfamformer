import pandas as pd
import numpy as np
import os

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