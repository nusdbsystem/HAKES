# %%
"""
Example of using HAKES for retrieval augmented generation

Service Setup

1. Run the search worker inside Docker

```shell
docker run --name search-worker-test -p 8080:8080 -v ./index:/mounted_store/index hakes-searchworker:v1
```

2. Run MongoDB inside Docker

```shell
docker run -d -p 27017:27017 --name mongo-test mongo
```

3. Configure environment variables

```shell
export SEARCH_WORKER_ADDR=http://host:port
export MONGO_ADDR=mongodb://host:port
export HF_API_KEY=api_key
```
"""

# %%
import os
import numpy as np
import random
import torch
from hakesclient import Client, DataType
from hakesclient.components.store import Store
from hakesclient.components.embedder import Embedder
from hakesclient.components.searcher import Searcher
from hakesclient.extensions.mongodb import MongoDB
from hakesclient.extensions.huggingface import HuggingFaceEmbedder
from hakes.index.build import (
    init_hakes_params,
    build_dataset,
    train_hakes_params,
    recenter_ivf,
)
from dotenv import load_dotenv

load_dotenv(".env")  # read .env file in the current directory

INDEX_DIR = "./collections"
COLLECTION_NAME = "poc"

# %%
# Initialize and configure HAKES client
store: Store = MongoDB(os.getenv("MONGO_ADDR"), "hakes", COLLECTION_NAME)
embedder: Embedder = HuggingFaceEmbedder(
    os.getenv("HF_API_KEY"), "google/embeddinggemma-300m"
)
searcher: Searcher = Searcher([os.getenv("SEARCH_WORKER_ADDR")])
client: Client = Client(
    embedder=embedder,
    store=store,
    searcher=searcher,
)

# %%
"""
Initialize a collection 
Each collection will builds an index based on its data distribution. Here, we use a random generated vector dataset for illustration.
"""


def init_index(collection_name):
    os.makedirs(os.path.join(INDEX_DIR, COLLECTION_NAME), exist_ok=True)
    os.makedirs(os.path.join(INDEX_DIR, COLLECTION_NAME, "checkpoint_0"), exist_ok=True)
    dir = os.path.join(INDEX_DIR, collection_name, "checkpoint_0")

    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    data = np.random.randn(10000, 768).astype(np.float32)
    data = data / np.linalg.norm(data, axis=1, keepdims=True)

    index = init_hakes_params(data, 384, 100, "ip")
    index.set_fixed_assignment(True)
    index.save_as_hakes_index(os.path.join(dir, "findex.bin"))
    sample_ratio = 0.1
    dataset = build_dataset(data, sample_ratio=sample_ratio, nn=50)
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
    recenter_ivf(index, data, sample_ratio)
    index.save_as_hakes_index(os.path.join(dir, "uindex.bin"))


# init index if index checkpoint files of specified collection not exist
if not (
    os.path.isdir(os.path.join(INDEX_DIR, COLLECTION_NAME))
    and any(os.scandir(os.path.join(INDEX_DIR, COLLECTION_NAME)))
):
    init_index(COLLECTION_NAME)
    # call load_collection twice cause container unexpectly exit, so assume collection is loaded if index file initialized
    client.load_collection(COLLECTION_NAME)

# %%
"""
Add data to the collection
HAKES allow continuously adding new data to the collection to enhance the coverage of application-specific information for RAG.
"""
from hakesclient.utils import texts_to_bytes

doc = {
    "key": "doc1",
    "value": "Bob and Alice first met in 1995 at a party in New York. They quickly became friends and started dating a year later. In 2000, they got married in a beautiful ceremony in Central Park. Over the years, they have traveled the world together, visiting places like Paris, Tokyo, and Sydney. They have two children, a son named Charlie and a daughter named Daisy. Bob works as a software engineer, while Alice is a graphic designer. They both enjoy hiking, cooking, and spending time with their family.",
}

result = client.add(
    COLLECTION_NAME, [doc["key"]], texts_to_bytes([doc["value"]]), DataType.TEXT, None
)
print(result)

# %%
"""
Perform RAG
HAKES retrieves relevant contexts to augment the user query to ground the LLM response generation.
"""
from hakesclient.utils import bytes_to_texts

question = "Where did Bob and Alice first meet?"

search_result = bytes_to_texts(
    client.search(
        COLLECTION_NAME, texts_to_bytes([question]), DataType.TEXT, 3, 20, 5, "IP"
    )
)

# generate prompt
context_info = ""
for i, d in enumerate(search_result):
    if not d:
        continue
    context_info += f"{i + 1}：{d}\n"
prompt = f"""
Please answer the following question based only on the context provided.\n
<Context>\n
{context_info}
</Context>\n
Question: {question}\n
Answer:
"""

print(prompt)
