from pfamformer.data_handling import EmbeddingDataset
from pfamformer.mlp import MLPClassifier
from torch.utils.data import DataLoader
import glob, os

mlp = MLPClassifier(960, 17929)

def create_set(set_type):
    embeddings = sorted(glob.glob(f'data/embeddings/{set_type}/*.npy'))[:1]
    label_names = [os.path.basename(e).split("_")[0] for e in embeddings]
    labels = [os.path.join(f"data/random_split/{set_type}/{name}") for name in label_names]
    return EmbeddingDataset(embeddings, labels)

train_set, test_set = create_set("train"), create_set("test")
train_batch = DataLoader(train_set, batch_size=128, shuffle=True)
mlp.train_model(train_batch, epochs=30, lr=1e-3)
