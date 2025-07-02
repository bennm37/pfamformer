import numpy as np
import matplotlib.pyplot as plt 
from pfamformer.data_handling import load, clean_train

train_df = clean_train(load("data/random_split/train.csv"))
grouped = train_df.groupby(by="accession_no").size()
nos, _ = zip(*sorted(grouped.items(), key=lambda x: x[1], reverse=True))
cumulative = np.cumsum(nos)
n_params = np.linspace(1, len(nos), len(nos)) * 961

fig, ax = plt.subplots()
ax.plot(n_params, cumulative)
plt.show()