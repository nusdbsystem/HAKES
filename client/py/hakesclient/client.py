import logging
import requests
import json
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from .cliconf import ClientConfig
from .message import (
    encode_search_worker_add_request,
    decode_search_worker_add_response,
    encode_search_worker_search_request,
    decode_search_worker_search_response,
    encode_search_worker_rerank_request,
    decode_search_worker_rerank_response,
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

    def search_worker_add(self, addr: str, vecs: np.ndarray, ids: np.ndarray):
        if len(addr) == 0:
            raise ValueError("search worker address is empty")
        if not addr.startswith("http"):
            addr = "http://" + addr
        data = encode_search_worker_add_request(vecs, ids)
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

    def search_worker_search(
        self,
        addr: str,
        query: np.ndarray,
        k: int,
        nprobe: int,
        k_factor: int,
        metric_type: str,
    ):
        if len(addr) == 0:
            raise ValueError("search worker address is empty")
        if not addr.startswith("http"):
            addr = "http://" + addr
        data = encode_search_worker_search_request(
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
        query: np.ndarray,
        k: int,
        input_ids: np.ndarray,
        nprobe: int,
        metric_type: str,
    ):
        if len(addr) == 0:
            raise ValueError("search worker address is empty")
        if not addr.startswith("http"):
            addr = "http://" + addr
        data = encode_search_worker_rerank_request(
            k,
            query,
            input_ids,
            nprobe,
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

    def add(self, vecs: np.ndarray, ids: np.ndarray):
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
            vecs,
            ids,
        )

    def search(
        self,
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
            addr, query, k, nprobe, k_factor, metric_type
        )

        if result is None:
            return None

        print(f"search result: {result}")

        print(query.shape)
        result = self.search_worker_rerank(
            addr, query, k, np.array(result["ids"]), nprobe, metric_type
        )

        return result

    def checkpoint(self):
        """
        Checkpoint the vector indexes (not implemented)
        """
        pass
