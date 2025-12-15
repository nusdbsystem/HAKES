"""Remote HTTP client for talking to hakes-server using bearer tokens.

This provides a drop-in simple wrapper that mirrors `Client` methods but
forwards requests to the `hakes-server` REST API and manages the access token.
"""
from typing import List, Optional
import requests
import numpy as np
import json

from .utils import ids_to_xids, xids_to_ids, bytes_to_texts


class RemoteClient:
    def __init__(self, server_addr: str, token: Optional[str] = None):
        if not server_addr.startswith("http"):
            server_addr = "http://" + server_addr
        self.server = server_addr.rstrip("/")
        self.token = token

    def _headers(self):
        h = {"Content-Type": "application/json"}
        if self.token:
            h["Authorization"] = f"Bearer {self.token}"
        return h

    def register(self, username: str, password: str):
        url = f"{self.server}/register"
        r = requests.post(url, json={"username": username, "password": password})
        r.raise_for_status()
        return r.json()

    def login(self, username: str, password: str):
        url = f"{self.server}/login"
        r = requests.post(url, json={"username": username, "password": password})
        r.raise_for_status()
        data = r.json()
        self.token = data.get("access_token")
        return data

    def load_collection(self, collection_name: str):
        url = f"{self.server}/load_collection"
        r = requests.post(url, json={"collection_name": collection_name}, headers=self._headers())
        r.raise_for_status()
        return r.json()

    def add(self, collection_name: str, keys: List[str], values: List[bytes], data_type: str, ids: Optional[np.ndarray] = None):
        # For simplicity send values as utf-8 strings for text data
        payload = {
            "collection_name": collection_name,
            "keys": keys,
            "values": [v.decode("utf-8") if isinstance(v, (bytes, bytearray)) else v for v in values],
            "data_type": data_type,
            "ids": None if ids is None else ids.tolist(),
        }
        url = f"{self.server}/add"
        r = requests.post(url, json=payload, headers=self._headers())
        r.raise_for_status()
        return r.json()

    def search(self, collection_name: str, query: bytes, data_type: str, k: int, nprobe: int, k_factor: int = 1, metric_type: str = "IP"):
        payload = {
            "collection_name": collection_name,
            "query": query.decode("utf-8") if isinstance(query, (bytes, bytearray)) else query,
            "data_type": data_type,
            "k": k,
            "nprobe": nprobe,
            "k_factor": k_factor,
            "metric_type": metric_type,
        }
        url = f"{self.server}/search"
        r = requests.post(url, json=payload, headers=self._headers())
        r.raise_for_status()
        return r.json()

    def delete(self, collection_name: str, ids: np.ndarray):
        # minimal delete: forward to /delete (server may expect JSON body adjustments)
        payload = {"collection_name": collection_name, "ids": ids.tolist()}
        url = f"{self.server}/delete"
        r = requests.post(url, json=payload, headers=self._headers())
        r.raise_for_status()
        return r.json()

    def checkpoint(self, collection_name: str):
        url = f"{self.server}/checkpoint"
        r = requests.post(url, json={"collection_name": collection_name}, headers=self._headers())
        r.raise_for_status()
        return r.json()
