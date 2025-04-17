import json
import logging
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests

from .cliconf import ClientConfig
from .message import (
    decode_search_worker_add_response,
    decode_search_worker_rerank_response,
    decode_search_worker_search_response,
    decode_search_worker_load_collection_response,
    encode_search_worker_add_request,
    encode_search_worker_rerank_request,
    encode_search_worker_search_request,
    encode_search_worker_load_collection_request,
    encode_search_worker_delete_request,
    decode_search_worker_delete_response,
)


class Client:
    def __init__(self, cfg: ClientConfig):
        self.cfg = cfg
        self.pool = ThreadPoolExecutor(max_workers=1000)
        # raise warning for distributed search worker
        if len(cfg.search_worker_addrs) > 1:
            logging.warning(
                f"Using distributed search worker with {len(cfg.search_worker_addrs)} workers not supported yet"
            )

    def search_work_load_collection(self, addr: str, collection_name: str):
        addr = self._validate_addr(addr)
        data = encode_search_worker_load_collection_request(collection_name)

        try:
            response = requests.post(addr + "/load", json=data)
        except Exception as e:
            logging.warning(f"search worker load collection failed on {addr}: {e}")
            return None

        if response.status_code != 200:
            logging.warning(
                f"Failed to call search worker, status code: {response.status_code} {response.text}"
            )
            return None

        return decode_search_worker_load_collection_response(json.loads(response.text))

    def _validate_addr(self, addr: str):
        if not addr:
            raise ValueError("search worker address is empty")

        if not addr.startswith("http"):
            addr = "http://" + addr

        return addr

    def search_worker_add(
        self, addr: str, collection_name: str, vecs: np.ndarray, ids: np.ndarray
    ):
        addr = self._validate_addr(addr)
        data = encode_search_worker_add_request(collection_name, vecs, ids)

        try:
            response = requests.post(addr + "/add", json=data)
        except Exception as e:
            logging.warning(f"search worker add failed on {addr}: {e}")
            return None

        if response.status_code != 200:
            logging.warning(
                f"Failed to call search worker, status code: {response.status_code} {response.text}"
            )
            return None

        return decode_search_worker_add_response(json.loads(response.text))

    def search_worker_delete(
        self, addr: str, collection_name: str, ids: np.ndarray
    ):
        addr = self._validate_addr(addr)
        data = encode_search_worker_delete_request(collection_name, ids)

        try:
            response = requests.post(addr + "/delete", json=data)
        except Exception as e:
            logging.warning(f"search worker delete failed on {addr}: {e}")
            return None

        if response.status_code != 200:
            logging.warning(
                f"Failed to call search worker, status code: {response.status_code} {response.text}"
            )
            return None

        return decode_search_worker_delete_response(json.loads(response.text))

    def search_worker_search(
        self,
        addr: str,
        collection_name: str,
        query: np.ndarray,
        k: int,
        nprobe: int,
        k_factor: int,
        metric_type: str,
    ):
        addr = self._validate_addr(addr)
        data = encode_search_worker_search_request(
            collection_name,
            k,
            query,
            nprobe,
            k_factor,
            metric_type,
        )

        try:
            response = requests.post(addr + "/search", json=data)
        except Exception as e:
            logging.warning(f"search worker search failed on {addr}: {e}")
            return None

        if response.status_code != 200:
            logging.warning(
                f"Failed to call search worker, status code: {response.status_code} {response.text}"
            )
            return None

        return decode_search_worker_search_response(
            json.loads(response.text), k * k_factor, False
        )

    def search_worker_rerank(
        self,
        addr: str,
        collection_name: str,
        query: np.ndarray,
        k: int,
        input_ids: np.ndarray,
        metric_type: str,
    ):
        addr = self._validate_addr(addr)
        data = encode_search_worker_rerank_request(
            collection_name,
            k,
            query,
            input_ids,
            metric_type,
        )

        try:
            response = requests.post(addr + "/rerank", json=data)
        except Exception as e:
            logging.warning(f"search worker rerank failed on {addr}: {e}")
            return None

        if response.status_code != 200:
            logging.warning(
                f"Failed to call search worker, status code: {response.status_code} {response.text}"
            )
            return None

        return decode_search_worker_rerank_response(json.loads(response.text), k)

    def load_collection(self, collection_name: str):
        """
        Load a collection to the distributed HakesService V3
        """
        addr = self.cfg.search_worker_addrs[0]
        return self.search_work_load_collection(addr, collection_name)

    def add(self, collection_name: str, vecs: np.ndarray, ids: np.ndarray):
        """
        Add vectors to the distributed HakesService V3
            1. add to the target refine index server
            2. add to all base index servers
        """
        # build vector batch for each client
        search_worker_addr = self.cfg.search_worker_addrs[0]

        # send requests to each server with the threadpool
        return self.search_worker_add(
            search_worker_addr,
            collection_name,
            vecs,
            ids,
        )

    def search(
        self,
        collection_name: str,
        query: np.ndarray,
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
        result = self.search_worker_search(
            addr, collection_name, query, k, nprobe, k_factor, metric_type
        )

        if result is None:
            return None

        print(f"search result: {result}")

        print(query.shape)
        result = self.search_worker_rerank(
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
        return self.search_worker_delete(addr, collection_name, ids)

    def checkpoint(self):
        """
        Checkpoint the vector indexes (not implemented)
        """
        pass
