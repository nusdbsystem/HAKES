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
import time
from typing import List

from utils import validate_addr
from message import (
    encode_list_collection_request,
    decode_list_collection_response,
    encode_load_collection_request,
    decode_load_collection_response,
    encode_add_request,
    decode_add_response,
    encode_search_request,
    decode_search_response,
    encode_delete_request,
    decode_delete_response,
    encode_checkpoint_request,
    decode_checkpoint_response,
    encode_rerank_request,
    decode_rerank_response,
)


class DataType:
    TEXT = "text"
    BINARY = "binary"


class Client:
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.addr = f"http://{host}:{port}"
        self.session = None
        self.connected = False
        self.token = None
        self.sub = None
        self.exp = None
        self.roles = None

    def connect(self, username: str, password: str):
        """Establish connection to the server with authentication."""
        if self.connected:
            logging.warning("Already connected")
            return True

        # Create a session for persistent connections
        self.session = requests.Session()

        # Send authentication request to server
        auth_data = {"username": username, "password": password}

        try:
            response = self.session.post(self.addr + "/login", json=auth_data)
            if response.status_code == 200:
                auth_response = json.loads(response.text)
                self.token = auth_response.get("access_token")
                self.sub = auth_response.get("sub")
                self.exp = auth_response.get("exp")
                self.roles = auth_response.get("roles")

                if self.token:
                    self.session.headers.update(
                        {"Authorization": f"Bearer {self.token}"}
                    )
                self.connected = True
                logging.info(
                    f"Successfully authenticated and connected to server at {self.addr}"
                )
                return True
            else:
                logging.error(
                    f"Authentication failed: {response.status_code} {response.text}"
                )
                return False
        except Exception as e:
            logging.error(f"Connection failed: {e}")
            return False

    def _check_token(self):
        """Check if token is expired."""
        if not self.exp or time.time() > self.exp:
            logging.error(
                "Authentication token has expired. Please reconnect by calling connect() again."
            )
            return False
        return True

    def _ensure_connected(self):
        """Ensure the client is connected before making requests."""
        if not self.connected:
            raise RuntimeError("Client is not connected. Call connect() first.")
        if not self._check_token():
            raise RuntimeError(
                "Authentication token has expired. Please reconnect by calling connect() again."
            )

    def close(self):
        """Close the connection and clean up sessions on both client and server."""
        if not self.connected:
            logging.warning("Client is not connected")
            return

        # Send logout request to server for session cleanup
        try:
            response = self.session.post(self.addr + "/logout")
            if response.status_code == 200:
                logging.info("Successfully logged out from server")
            else:
                logging.warning(
                    f"Logout request failed: {response.status_code} {response.text}"
                )
        except Exception as e:
            logging.warning(f"Failed to send logout request: {e}")

        # Clean up local session
        if self.session:
            self.session.close()

        # Reset connection state
        self.session = None
        self.connected = False
        self.token = None
        self.sub = None
        self.exp = None
        self.roles = None
        logging.info("Local session cleaned up")

    def list_collections(self):
        self._ensure_connected()
        data = encode_list_collection_request(user_id=self.sub, token=self.token)

        try:
            response = self.session.post(self.addr + "/list", json=data)
        except Exception as e:
            logging.warning(f"search worker list collection failed on {self.addr}: {e}")
            return None

        if response.status_code != 200:
            logging.warning(
                f"Failed to call search worker, status code: {response.status_code} {response.text}"
            )
            return None

        return decode_list_collection_response(json.loads(response.text))

    def load_collection(self, collection_name: str):
        self._ensure_connected()
        data = encode_load_collection_request(
            collection_name, user_id=self.sub, token=self.token
        )

        try:
            response = self.session.post(self.addr + "/load", json=data)
        except Exception as e:
            logging.warning(f"search worker load collection failed on {self.addr}: {e}")
            return None

        if response.status_code != 200:
            logging.warning(
                f"Failed to call search worker, status code: {response.status_code} {response.text}"
            )
            return None

        return decode_load_collection_response(json.loads(response.text))

    def add(
        self,
        collection_name: str,
        keys: List[str],
        values: List[bytes],
        data_type: DataType,
        ids: np.ndarray | None,
    ):
        self._ensure_connected()
        data = encode_add_request(
            collection_name,
            json.dumps(keys),
            json.dumps(values),
            data_type,
            ids,
            user_id=self.sub,
            token=self.token,
        )

        try:
            response = self.session.post(self.addr + "/add", json=data)
        except Exception as e:
            logging.warning(f"search worker add failed on {self.addr}: {e}")
            return None

        if response.status_code != 200:
            logging.warning(
                f"Failed to call search worker, status code: {response.status_code} {response.text}"
            )
            return None

        return decode_add_response(json.loads(response.text))

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
        self._ensure_connected()
        data = encode_search_request(
            collection_name,
            query,
            data_type,
            k,
            nprobe,
            k_factor,
            metric_type,
            user_id=self.sub,
            token=self.token,
        )

        try:
            response = self.session.post(self.addr + "/search", json=data)
        except Exception as e:
            logging.warning(f"search worker search failed on {self.addr}: {e}")
            return None

        if response.status_code != 200:
            logging.warning(
                f"Failed to call search worker, status code: {response.status_code} {response.text}"
            )
            return None

        return decode_search_response(json.loads(response.text), k)

    def delete(self, collection_name: str, ids: np.ndarray):
        self._ensure_connected()
        data = encode_delete_request(collection_name, ids, token=self.token)

        try:
            response = self.session.post(self.addr + "/delete", json=data)
        except Exception as e:
            logging.warning(f"search worker delete failed on {self.addr}: {e}")
            return None

        if response.status_code != 200:
            logging.warning(
                f"Failed to call search worker, status code: {response.status_code} {response.text}"
            )
            return None

        return decode_delete_response(json.loads(response.text))

    def checkpoint(self, collection_name: str):
        self._ensure_connected()
        data = encode_checkpoint_request(collection_name, token=self.token)

        try:
            response = self.session.post(self.addr + "/checkpoint", json=data)
        except Exception as e:
            logging.warning(f"search worker checkpoint failed on {self.addr}: {e}")
            return None

        if response.status_code != 200:
            logging.warning(
                f"Failed to call search worker, status code: {response.status_code} {response.text}"
            )
            return None

        return decode_checkpoint_response(json.loads(response.text))

    def rerank(
        self,
        collection_name: str,
        query: np.ndarray,
        k: int,
        input_ids: np.ndarray,
        metric_type: str,
    ):
        self._ensure_connected()
        data = encode_rerank_request(
            collection_name,
            k,
            query,
            input_ids,
            metric_type,
            user_id=self.sub,
            token=self.token,
        )

        try:
            response = self.session.post(self.addr + "/rerank", json=data)
        except Exception as e:
            logging.warning(f"search worker rerank failed on {self.addr}: {e}")
            return None

        if response.status_code != 200:
            logging.warning(
                f"Failed to call search worker, status code: {response.status_code} {response.text}"
            )
            return None

        return decode_rerank_response(json.loads(response.text), k)
