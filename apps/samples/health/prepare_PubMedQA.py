# %%
# %load_ext autoreload
# %autoreload 2
# %%
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from datasets import load_dataset

data = load_dataset("qiaojin/PubMedQA", "pqa_artificial")

# %%
# check the data
print(data.keys())
print(len(data["train"]))
print(data["train"][0])


# The format is in JSON: {"pubid": 12345678, "question": "...", "context": {"contexts": [], "labels":[], "meshes": []}, "long_answer": "...", "final_decision": "yesno"}
# %%
# transform the data to keep the pubid and then merge the question contexts and long_answer into one string
def transform(example):
    content = (
        example["question"]
        + "\n"
        + "\n".join(example["context"]["contexts"])
        + "\n"
        + example["long_answer"]
    )
    return {"id": example["pubid"], "text": content}


processed_data = data.map(
    transform,
    remove_columns=["pubid", "question", "context", "long_answer", "final_decision"],
)

# %%
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
model = SentenceTransformer(
    "nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True, device=device
)
sentences = ["hello world"]
embeddings = model.encode(sentences)
print(embeddings)  # not normalized
# normalize
embeddings = F.normalize(torch.tensor(embeddings), p=2, dim=1)
print(embeddings)


# %%
# process all the data in the data in the processed_data to add a field for vector
def embed(example):
    sentences = [example["text"]]
    embeddings = model.encode(sentences)
    embeddings = F.normalize(torch.tensor(embeddings), p=2, dim=1)
    return {"vector": embeddings[0].numpy()}


# %%
processed_data = processed_data.map(embed)
print(processed_data)
# %%
processed_data.save_to_disk("pubmedqa_embedded")
# %%
from datasets import load_from_disk

processed_data = load_from_disk("pubmedqa_embedded")
# %%
# prepare index files
import os
import numpy as np
import random
import torch
from hakes.index.build import (
    init_hakes_params,
    build_dataset,
    train_hakes_params,
    recenter_ivf,
)


INDEX_DIR = "./collections"
COLLECTION_NAME = "pubmedqa"

# %%
print(processed_data)
print(processed_data["train"]["vector"])
# get vector as numpy array
data = np.array(processed_data["train"]["vector"], dtype=np.float32)
print(data.shape)
print(type(data[0][0]))


# %%
def init_index(collection_name, data, vt_out, nlist=512):
    os.makedirs(os.path.join(INDEX_DIR, COLLECTION_NAME), exist_ok=True)
    os.makedirs(os.path.join(INDEX_DIR, COLLECTION_NAME, "checkpoint_0"), exist_ok=True)
    dir = os.path.join(INDEX_DIR, collection_name, "checkpoint_0")

    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    data = data / np.linalg.norm(data, axis=1, keepdims=True)

    index = init_hakes_params(data, vt_out, nlist, "ip")
    index.set_fixed_assignment(True)
    index.save_as_hakes_index(os.path.join(dir, "findex.bin"))
    sample_ratio = 0.1
    dataset = build_dataset(data, sample_ratio=sample_ratio, nn=50)
    train_hakes_params(
        model=index,
        dataset=dataset,
        epochs=2,
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
    recenter_ivf(index, data, sample_ratio)
    index.save_as_hakes_index(os.path.join(dir, "uindex.bin"))


init_index(COLLECTION_NAME, data, 192, 512)
