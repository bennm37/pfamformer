import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
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
        dev_dataloader=None,
        epochs=10,
        lr=1e-3,
        patience=3,
        device='cpu',
        save=True,
    ):
        print(f"Started training.")
        self.device = device
        self.to(self.device)
        optimizer = optim.Adam(self.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss()
        self.train_metrics = pd.DataFrame(columns=["epoch","loss","accuracy","precision","recall","f1"])
        self.dev_dataloader = dev_dataloader
        if dev_dataloader is not None:
            self.dev_metrics = pd.DataFrame(columns=["epoch", "accuracy","precision","recall","f1"])
        early_counter = 0
        prev_loss = np.inf
        for epoch in range(epochs):
            print(f"Starting Epoch {epoch}/{epochs}")
            self.train()
            epoch_loss = 0
            all_preds = []
            all_labels = []
            for X, y in tqdm(dataloader):
                X = X.to(self.device)
                y = y.to(self.device)
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
            self.train_metrics = pd.concat([self.train_metrics, df])
            print(f"Train Metrics ...")
            print(f"Loss = {loss}")
            self.log(df, epochs)
            self.update_plot()
            if self.dev_dataloader is not None:
                dev_df = self.evalute_dev()
                dev_df["epoch"] = epoch
                self.dev_metrics = pd.concat([self.dev_metrics, dev_df])
                print(f"Dev Metrics ...")
                self.log(dev_df, epochs)
            if prev_loss < epoch_loss:
                early_counter += 1
            if early_counter >= patience:
                print(f"Stopping Early at Epoch {epoch} due to early_counter exceeding patience {patience}.")
                break
            prev_loss = epoch_loss
        if save:
            self.save_model()
        return self.train_metrics
            
    def evalute_dev(self):
        all_preds = []
        all_labels = []
        for X, y in self.dev_dataloader:
            X = X.to(self.device)
            y = y.to(self.device)
            logits = self(X)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.cpu().numpy())
        df = compute_metrics(all_labels, all_preds)
        return df

    def save_model(self, save_folder="data/trained"):
        dt = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"{save_folder}/mlp_{dt}.pkl"
        with open(filename, "wb") as f:
            pickle.dump(self, f)


    def update_plot(self):
        mean_metrics = self.train_metrics.groupby("epoch").apply(np.mean, axis=0, include_groups=False)
        std_metrics = self.train_metrics.groupby("epoch").apply(np.std, axis=0, include_groups=False)
        epochs = self.train_metrics["epoch"].unique()
        if not hasattr(self, '_fig'):
            self._fig, self._axs = plt.subplots(2, 2, figsize=(6,6))
            self._lines = {}
            self._lines["loss"] = self._axs[0,0].plot(epochs, mean_metrics["loss"])[0]
            plt.tight_layout()
            plt.savefig(f"media")
        else:
            self._lines["loss"].set_data(epochs, mean_metrics["loss"])
            self._axs[0,0].relim()
            self._axs[0,0].autoscale_view()
            self._fig.canvas.draw()
            self._fig.canvas.flush_events()
            plt.draw()


    def log(self, df, epochs):
        print(f"Finished Epoch {df['epoch'].values[0]}/{epochs}")
        for metric in ["accuracy","precision","recall", "f1"]:
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
