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

import logging
import numpy as np

from concurrent.futures import ThreadPoolExecutor
from .cliconf import ClientConfig
from .components.searcher import Searcher


class Client:
    def __init__(self, cfg: ClientConfig):
        self.cfg = cfg
        self.pool = ThreadPoolExecutor(max_workers=1000)
        # raise warning for distributed search worker
        if len(cfg.search_worker_addrs) > 1:
            logging.warning(
                f"Using distributed search worker with {len(cfg.search_worker_addrs)} workers not supported yet"
            )
        self.searcher = Searcher()

    def load_collection(self, collection_name: str):
        """
        Load a collection to the distributed HakesService V3
        """
        addr = self.cfg.search_worker_addrs[0]
        return self.searcher.load_collection(addr, collection_name)

    def add(self, collection_name: str, vecs: np.ndarray, ids: np.ndarray):
        """
        Add vectors to the distributed HakesService V3
            1. add to the target refine index server
            2. add to all base index servers
        """
        # build vector batch for each client
        search_worker_addr = self.cfg.search_worker_addrs[0]

        # send requests to each server with the threadpool
        return self.searcher.add(
            search_worker_addr,
            collection_name,
            vecs,
            ids,
        )

    def search(
        self,
        collection_name: str,
        query: np.ndarray | str,
        k: int,
        nprobe: int,
        k_factor: int = 1,
        metric_type: str = "IP",
    ):
        """
        Search vectors in the distributed HakesService V3
        """
        print(query.shape)
        if len(query.shape) != 2:
            logging.warning(f"search failed: query shape {query.shape} != 2")
            return None

        addr = self.cfg.search_worker_addrs[0]
        result = self.searcher.search(
            addr, collection_name, query, k, nprobe, k_factor, metric_type
        )

        if result is None:
            return None

        print(f"search result: {result}")

        print(query.shape)
        result = self.searcher.rerank(
            addr,
            collection_name,
            query,
            k,
            np.array(result["ids"]),
            metric_type,
        )

        return result

    def delete(self, collection_name: str, ids: np.ndarray):
        """
        Delete vectors from the distributed HakesService V3
        """
        addr = self.cfg.search_worker_addrs[0]
        return self.searcher.delete(addr, collection_name, ids)

    def checkpoint(self, collection_name: str):
        """
        Checkpoint the vector indexes (not implemented)
        """
        addr = self.cfg.search_worker_addrs[0]
        return self.searcher.checkpoint(addr, collection_name)
