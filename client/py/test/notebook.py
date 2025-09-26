# %%
%load_ext autoreload
%autoreload 2
# %%
import numpy as np
import sys
sys.path.append('..')
import numpy as np
import random

# %%
COLLECTION_NAME = "sample_collection"

# %%
# fix all randomness
seed = 0
random.seed(seed)
np.random.seed(seed)

# %%
dim = 1024
database_vector = np.random.randn(1000, dim).astype(np.float32)
database_vector = database_vector / np.linalg.norm(database_vector, axis=1, keepdims=True)
query_vector = np.random.randn(100, dim).astype(np.float32)
query_vector = query_vector / np.linalg.norm(query_vector, axis=1, keepdims=True)

# %%
from hakesclient import Client, ClientConfig
# %%
config = ClientConfig(search_worker_addrs=["http://localhost:2351"])
client = Client(config)
# %%
client.load_collection(COLLECTION_NAME)
# %%
for i in range(len(database_vector)):
    client.add(COLLECTION_NAME, database_vector[i:i+1], [i])

# %%
client.search(COLLECTION_NAME, query_vector[0:1], 10, 20, 5, "IP")
# %%
client.delete(COLLECTION_NAME, [439, 444])
# %%
client.search(COLLECTION_NAME, query_vector[0:1], 10, 20, 5, "IP")
# %%
client.checkpoint(COLLECTION_NAME)
# %%
