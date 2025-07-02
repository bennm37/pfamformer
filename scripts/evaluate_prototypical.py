from pfamformer.prototypical import PrototypicalClassifier
from train_mlp import create_pfam
import tqdm
from pfamformer.mlp import compute_metrics

train_set, dev_set, test_set = create_pfam()
pc = PrototypicalClassifier(train_set, "cuda", "cosine")
preds = []
labels = []
for x, y in tqdm(dev_set, desc="Evaluating Dev"):
  preds.append(pc(x.reshape((1, 960))).to("cpu").item())
  labels.append(y.to("cpu").item())
print(compute_metrics(labels, preds).mean(axis=0).mean(axis=0))
