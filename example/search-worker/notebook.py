# %%
"""
Launch the search-worker first

docker run --name search-worker-test -p 2053:8080 -v $PWD/../gen-index/sample_data:/mounted_store/index hakes-searchworker-nosgx:v1

"""
# %%
%load_ext autoreload
%autoreload 2
# %%
import sys
sys.path.append(os.path.abspath("../../client/py"))
print(sys.path)
# %%
import numpy as np
import random

# %%
OUTPUT_DIR = "../gen-index/sample_data"

# %%
# fix all randomness
seed = 0
random.seed(seed)
np.random.seed(seed)

# %%
# generate normalized vectors
database_vector = np.random.randn(10000, 768).astype(np.float32)
database_vector = database_vector / np.linalg.norm(database_vector, axis=1, keepdims=True)
query_vector = np.random.randn(1000, 768).astype(np.float32)
query_vector = query_vector / np.linalg.norm(query_vector, axis=1, keepdims=True)

# %%
from hakesclient import Client, ClientConfig

# %%
config = ClientConfig(search_worker_addrs=["http://localhost:2053"])
client = Client(config)
# %%
for i in range(len(database_vector)):
    client.add(database_vector[i:i+1], [i])

# %%
client.search(query_vector[0:1], 10, 20, 5, "IP")

# %%
