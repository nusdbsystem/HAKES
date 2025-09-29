# Copyright 2024 The HAKES Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List
import numpy as np

from concurrent.futures import ThreadPoolExecutor
from .components.searcher import Searcher
from .components.embedder import Embedder
from .components.store import Store
from .utils import ids_to_xids, xids_to_ids, bytes_to_texts

class DataType:
    TEXT = 'text'
    BINARY = 'binary'

class Client:
    def __init__(self, store: Store = None, embedder: Embedder = None, searcher: Searcher = None):
        self.embedder = embedder
        self.store = store
        self.pool = ThreadPoolExecutor(max_workers=1000)
        self.searcher = searcher

    def load_collection(self, collection_name: str):
        """
        Load a collection to the distributed HakesService V3
        """
        return self.searcher.load_collection(collection_name)

    def add(self, collection_name: str, keys: List[str], values: List[bytes], data_type: DataType, ids: np.ndarray | None):
        """
        Add vectors to the distributed HakesService V3
            1. add to the target refine index server
            2. add to all base index servers
        """
        # embed the data
        if data_type == DataType.TEXT:
            vectors = self.embedder.embed_text(bytes_to_texts(values))
        elif data_type == DataType.BINARY:
            vectors = self.embedder.embed_binary(values)
        else:
            raise ValueError(f"Unsupported data type: {data_type}")
        
        # store the original data and get xids
        success, xids = self.store.put(keys, values, None if ids == None else ids_to_xids(ids))
        if not success:
            raise RuntimeError("Failed to add data to the store")
        ids = xids_to_ids(xids)
        
        # send requests to each server with the threadpool
        return self.searcher.add(
            collection_name,
            vectors,
            ids,
        )

    def search(
        self,
        collection_name: str,
        query: bytes,
        data_type: DataType,
        k: int,
        nprobe: int,
        k_factor: int = 1,
        metric_type: str = "IP",
    ):
        """
        Search vectors in the distributed HakesService V3
        """

        # embed the query
        if data_type == DataType.TEXT:
            query_vec = self.embedder.embed_text(bytes_to_texts(query))
        elif data_type == DataType.BINARY:
            query_vec = self.embedder.embed_binary(query)
        else:
            raise ValueError(f"Unsupported data type: {data_type}")

        # search the query
        result = self.searcher.search(
            collection_name, query_vec, k, nprobe, k_factor, metric_type
        )
        if result is None:
            return None
        result = self.searcher.rerank(
            collection_name,
            query_vec,
            k,
            np.array(result["ids"]),
            metric_type,
        )

        # retrieve the original data from the store
        ids = result['ids'][0]
        xids = ids_to_xids(ids)
        data = self.store.get_by_ids(xids)

        return data

    def delete(self, collection_name: str, ids: np.ndarray):
        """
        Delete vectors from the distributed HakesService V3
        """
        # delete original data from the store
        xids = ids_to_xids(ids)
        success = self.store.delete_by_ids(xids)
        if not success:
            raise RuntimeError("Failed to delete data from the store")

        # delete vectors from the index
        return self.searcher.delete(collection_name, ids)

    def checkpoint(self, collection_name: str):
        """
        Checkpoint the vector indexes (not implemented)
        """
        return self.searcher.checkpoint(collection_name)
