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

import json
import logging
import numpy as np
import requests
from typing import List

from ..message import (
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
    encode_search_worker_checkpoint_request,
    decode_search_worker_checkpoint_response,
)
from ..utils import validate_addr


class Searcher:

    def __init__(self, addrs: List[str]):
        if len(addrs) > 1:
            logging.warning(
                f"Using distributed search worker with {len(addrs)} workers not supported yet"
            )
        validate_addr(addrs[0])
        self.addr = addrs[0]

    def load_collection(self, collection_name: str):
        data = encode_search_worker_load_collection_request(collection_name)

        try:
            response = requests.post(self.addr + "/load", json=data)
        except Exception as e:
            logging.warning(f"search worker load collection failed on {self.addr}: {e}")
            return None

        if response.status_code != 200:
            logging.warning(
                f"Failed to call search worker, status code: {response.status_code} {response.text}"
            )
            return None

        return decode_search_worker_load_collection_response(json.loads(response.text))

    def add(self, collection_name: str, vecs: np.ndarray, ids: np.ndarray):
        data = encode_search_worker_add_request(collection_name, vecs, ids)

        try:
            response = requests.post(self.addr + "/add", json=data)
        except Exception as e:
            logging.warning(f"search worker add failed on {self.addr}: {e}")
            return None

        if response.status_code != 200:
            logging.warning(
                f"Failed to call search worker, status code: {response.status_code} {response.text}"
            )
            return None

        return decode_search_worker_add_response(json.loads(response.text))

    def delete(self, collection_name: str, ids: np.ndarray):
        data = encode_search_worker_delete_request(collection_name, ids)

        try:
            response = requests.post(self.addr + "/delete", json=data)
        except Exception as e:
            logging.warning(f"search worker delete failed on {self.addr}: {e}")
            return None

        if response.status_code != 200:
            logging.warning(
                f"Failed to call search worker, status code: {response.status_code} {response.text}"
            )
            return None

        return decode_search_worker_delete_response(json.loads(response.text))

    def search(
        self,
        collection_name: str,
        query: np.ndarray,
        k: int,
        nprobe: int,
        k_factor: int,
        metric_type: str,
    ):
        data = encode_search_worker_search_request(
            collection_name,
            k,
            query,
            nprobe,
            k_factor,
            metric_type,
        )

        try:
            response = requests.post(self.addr + "/search", json=data)
        except Exception as e:
            logging.warning(f"search worker search failed on {self.addr}: {e}")
            return None

        if response.status_code != 200:
            logging.warning(
                f"Failed to call search worker, status code: {response.status_code} {response.text}"
            )
            return None

        return decode_search_worker_search_response(
            json.loads(response.text), k * k_factor, False
        )

    def rerank(
        self,
        collection_name: str,
        query: np.ndarray,
        k: int,
        input_ids: np.ndarray,
        metric_type: str,
    ):
        data = encode_search_worker_rerank_request(
            collection_name,
            k,
            query,
            input_ids,
            metric_type,
        )

        try:
            response = requests.post(self.addr + "/rerank", json=data)
        except Exception as e:
            logging.warning(f"search worker rerank failed on {self.addr}: {e}")
            return None

        if response.status_code != 200:
            logging.warning(
                f"Failed to call search worker, status code: {response.status_code} {response.text}"
            )
            return None

        return decode_search_worker_rerank_response(json.loads(response.text), k)

    def checkpoint(self, collection_name: str):
        data = encode_search_worker_checkpoint_request(collection_name)

        try:
            response = requests.post(self.addr + "/checkpoint", json=data)
        except Exception as e:
            logging.warning(f"search worker load collection failed on {self.addr}: {e}")
            return None

        if response.status_code != 200:
            logging.warning(
                f"Failed to call search worker, status code: {response.status_code} {response.text}"
            )
            return None

        return decode_search_worker_checkpoint_response(json.loads(response.text))
