from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig
from pfamformer.data_handling import load, clean_train

train = clean_train(load("data/random_split/train"))
client = ESMC.from_pretrained("esmc_300m").to("cpu") # or "cpu"
for i in range(100):
    protein = ESMProtein(sequence="AAAAA")
    protein_tensor = client.encode(protein)
    logits_output = client.logits(
    protein_tensor, LogitsConfig(sequence=True, return_embeddings=True)
    )
    print(logits_output.logits, logits_output.embeddings)