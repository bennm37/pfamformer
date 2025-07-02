import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from collections import defaultdict

class PrototypicalClassifier(nn.Module):
    def __init__(self, train_loader, device, distance='euclidean'):
        super().__init__()
        self.device = device
        self.distance = distance
        self.class_prototypes = self._compute_prototypes(train_loader)

    def _compute_prototypes(self, loader):
        class_embeddings = defaultdict(list)
        for x, y in tqdm(loader, desc="Computing prototypes"):
            x = x.to(self.device)
            y = y.to(self.device)
            for cls in torch.unique(y):
                mask = y == cls
                class_embeddings[cls.item()].append(x[mask])
        prototypes = {}
        for cls, embeds in class_embeddings.items():
            embeds = torch.cat(embeds, dim=0)
            prototypes[cls] = embeds.mean(dim=0)
        class_ids = sorted(prototypes.keys())
        proto_tensor = torch.stack([prototypes[c] for c in class_ids])
        self.register_buffer("prototypes", proto_tensor)
        self.class_ids = torch.tensor(class_ids, device=self.device)
        return proto_tensor

    def forward(self, query_x):
        query_x = query_x.to(self.device)
        if self.distance == 'euclidean':
            dists = torch.cdist(query_x, self.prototypes)
        elif self.distance == 'cosine':
            query_x = F.normalize(query_x, dim=-1)
            prototypes = F.normalize(self.prototypes, dim=-1)
            dists = 1 - torch.matmul(query_x, prototypes.T)
        else:
            raise ValueError("Unsupported distance metric")
        logits = -dists
        probs = F.softmax(logits, dim=-1)
        return self.class_ids[probs.argmax()]
