# %%
import sys
import os
sys.path.append(os.path.abspath("../../python"))
print(sys.path)
# %%
import numpy as np
import random
import torch
from hakes.index.build import init_hakes_params, build_dataset, train_hakes_params, recenter_ivf

# %%
OUTPUT_DIR = "sample_data"
COLLECTION_NAME = "test_collection"

collection_dir = os.path.join(OUTPUT_DIR, COLLECTION_NAME)
os.makedirs(collection_dir, exist_ok=True)

# %%
# fix all randomness
seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# %%
# generate 10000 768 dimensional vectors
data = np.random.randn(10000, 768).astype(np.float32)
data = data / np.linalg.norm(data, axis=1, keepdims=True)
# %%
index = init_hakes_params(data, 384, 100, "ip")

# %%
index.save_as_hakes_index(os.path.join(collection_dir, "findex.bin"))

# %%
sample_ratio = 1
sample_ratio = 0.1
dataset = build_dataset(data, sample_ratio=sample_ratio, nn=50)

# %%
train_hakes_params(
    model=index,
    dataset=dataset,
    epochs=3,
    batch_size=128,
    lr_params={"vt": 1e-4, "pq": 1e-4, "ivf": 0},
    loss_weight={
        "vt": "rescale",
        "pq": 1,
        "ivf": 0,
    },
    temperature=1,
    loss_method="hakes",
    device="cpu",
)

# %%
recenter_ivf(index, data, sample_ratio)
index.save_as_hakes_index(os.path.join(collection_dir, "uindex.bin"))

# %%
