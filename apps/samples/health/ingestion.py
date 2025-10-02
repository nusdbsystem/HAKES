# %%
# %load_ext autoreload
# %autoreload 2
# %%
"""
launch the search worker and the store for data insertion

docker run --name search-worker-test -p 2053:8080 -v ./collections:/mounted_store/index hakes-searchworker:v1
docker run -d -p 27017:27017 --name mongo-test mongo
"""

# Initialize and configure HAKES client
from hakesclient import Client, DataType
from hakesclient.components.store import Store
from hakesclient.components.embedder import Embedder
from hakesclient.components.searcher import Searcher
from hakesclient.extensions.mongodb import MongoDB
from hakesclient.extensions.ollama import OllamaEmbedder

COLLECTION_NAME = "pubmedqa"

store: Store = MongoDB("mongodb://localhost:27017", "collections", COLLECTION_NAME)
embedder: Embedder = OllamaEmbedder(
    base_url="http://localhost:11500", model="nomic-embed-text"
)
searcher: Searcher = Searcher(["http://localhost:2053"])
client: Client = Client(
    embedder=embedder,
    store=store,
    searcher=searcher,
)
client.load_collection(COLLECTION_NAME)

# %%
# add data to the collection
from datasets import load_from_disk

processed_data = load_from_disk("pubmedqa_embedded")
print(processed_data)
# %%
# print(type(processed_data['train'][100]["id"]))
print(type(processed_data["train"][100]["vector"]))

# %%
for i in range(len(processed_data["train"])):
# for i in range(20):
    example = processed_data["train"][i]
    key = example["id"]
    value = example["text"].encode("utf-8")
    print(f"Adding {i} key={key} value={value}")
    res = client.add(
        COLLECTION_NAME,
        [key],
        [value],
        DataType.TEXT,
        None,
    )
    print(f"Result: {res}")

# %%
# checkpoint
client.checkpoint(COLLECTION_NAME)
