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
    def load_collection(self, addr: str, collection_name: str):
        addr = validate_addr(addr)
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

    def add(self, addr: str, collection_name: str, vecs: np.ndarray, ids: np.ndarray):
        addr = validate_addr(addr)
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

    def delete(self, addr: str, collection_name: str, ids: np.ndarray):
        addr = validate_addr(addr)
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

    def search(
        self,
        addr: str,
        collection_name: str,
        query: np.ndarray,
        k: int,
        nprobe: int,
        k_factor: int,
        metric_type: str,
    ):
        addr = validate_addr(addr)
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

    def rerank(
        self,
        addr: str,
        collection_name: str,
        query: np.ndarray,
        k: int,
        input_ids: np.ndarray,
        metric_type: str,
    ):
        addr = validate_addr(addr)
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

    def checkpoint(self, addr: str, collection_name: str):
        addr = validate_addr(addr)
        data = encode_search_worker_checkpoint_request(collection_name)

        try:
            response = requests.post(addr + "/checkpoint", json=data)
        except Exception as e:
            logging.warning(f"search worker load collection failed on {addr}: {e}")
            return None

        if response.status_code != 200:
            logging.warning(
                f"Failed to call search worker, status code: {response.status_code} {response.text}"
            )
            return None

        return decode_search_worker_checkpoint_response(json.loads(response.text))
