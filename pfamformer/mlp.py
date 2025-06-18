import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import matplotlib.pyplot as plt
import pandas as pd
import pickle 
from datetime import datetime
from tqdm import tqdm

class MLPClassifier(nn.Module):
    def __init__(self, embedding_size, num_labels, hidden_sizes=[]):
        super().__init__()
        self.layers = []
        in_dim = embedding_size
        for h in hidden_sizes:
            self.layers.append(nn.Linear(in_dim, h))
            self.layers.append(nn.ReLU())
            in_dim = h
        self.layers.append(nn.Linear(in_dim, num_labels))
        # self.layers.append(nn.Softmax(dim=0))
        self.layers.append(nn.Sigmoid())
        self.model = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.model(x)

    def train_model(
        self,
        dataloader,
        epochs=10,
        lr=1e-3,
        device='cpu',
        save=True,
    ):
        print(f"Started training.")
        self.to(device)
        optimizer = optim.Adam(self.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss()
        metrics = []
        for epoch in range(epochs):
            print(f"Starting Epoch {epoch}/{epochs}")
            self.train()
            epoch_loss = 0
            all_preds = []
            all_labels = []
            for X, y in tqdm(dataloader):
                X = X.to(device)
                y = y.to(device)
                optimizer.zero_grad()
                logits = self(X)
                loss = loss_fn(logits, y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * X.size(0)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(y.cpu().numpy())
            epoch_loss /= len(dataloader.dataset)
            df = compute_metrics(all_labels, all_preds)
            df["epoch"] = epoch
            df["loss"] = epoch_loss
            metrics.append(df)
            self.log(df, epochs)
        self.metrics = pd.concat(metrics)
        if save:
            self.save_model()
        return self.metrics
    
    def save_model(self, save_folder="data/trained"):
        dt = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"{save_folder}/mlp_{dt}.pkl"
        with open(filename, "wb") as f:
            pickle.dump(self, f)


    def update_plot(self, metrics, losses, num_labels):
        if not hasattr(self, '_fig'):
            self._fig, self._axs = plt.subplots(1, 4, figsize=(20, 4))
            self._lines = {}
            for i, metric in enumerate(['precision', 'recall', 'f1']):
                self._axs[i].set_title(f'{metric.capitalize()} per Class')
                self._axs[i].set_xlabel('Epoch')
                self._axs[i].set_ylabel(metric.capitalize())
                for label in range(num_labels):
                    (line,) = self._axs[i].plot([], [], label=f'Class {label}')
                    self._lines[(metric, label)] = line
                self._axs[i].legend()
            self._axs[3].set_title('Training Loss')
            self._axs[3].set_xlabel('Epoch')
            self._axs[3].set_ylabel('Loss')
            (self._loss_line,) = self._axs[3].plot([], [], label='Loss')
            plt.tight_layout()
            plt.ion()
            plt.show()

    def log(self, df, epochs):
        print(f"Finished Epoch {df['epoch'].values[0]/epochs}")
        for metric in ["accuracy","precision","recall","f1"]:
            print(f"{metric.capitalize()}: {df[metric].values.mean()} +- {df[metric].values.std()}")

    


def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average=None, zero_division=0)
    rec = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    df = pd.DataFrame({
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1
    })
    return df
