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
import os
from typing import List, Optional


class ClientConfig:
    def __init__(
        self,
        search_worker_addrs: Optional[List[str]] = None,
        preferred_search_worker: int = 0,
        hakes_addr: str = "",
        embed_endpoint_type: str = "",
        embed_endpoint_config: str = "",
        store_type: str = "",
        store_addr: str = "",
    ):
        self.search_worker_addrs = search_worker_addrs or []
        self.preferred_search_worker = preferred_search_worker
        self.hakes_addr = hakes_addr
        self.embed_endpoint_type = (
            embed_endpoint_type if embed_endpoint_type and embed_endpoint_type != "none" else ""
        )
        self.embed_endpoint_config = embed_endpoint_config
        self.store_type = (
            store_type if store_type and store_type != "none" and store_addr else ""
        )
        self.store_addr = (
            store_addr if store_type and store_type != "none" and store_addr else ""
        )
        self.n = len(self.search_worker_addrs)

    @staticmethod
    def from_file(path: str) -> "ClientConfig":
        """Read client config from file. The file must contains ALL needed fields.

        If path is not specified, the default path will be used: `$HOME/.hakes/client_config.json`.

        Args:
            path (str): config file path

        Returns:
            ClientConfig: The client config
        """
        if not path:
            # load the default path if not specified: $HOME/.hakes/client_config.json
            path = os.path.join(os.path.expanduser("~"), ".hakes", "config.json")

        # load from a json file
        with open(path, "r") as f:
            data = json.load(f)

        try:
            conf = ClientConfig(
                data["search_worker_addrs"],
                data["preferred_search_worker"],
                data["hakes_addr"],
                data["embed_endpoint_type"],
                data["embed_endpoint_config"],
                data["store_type"],
                data["store_addr"],
            )
        except KeyError as e:
            raise ValueError(f"Missing field {e} in {path}")

        return conf

    def save(self, path: str):
        # save as a json file
        with open(path, "w") as f:
            json.dump(self.to_dict(), f)

    def __repr__(self) -> str:
        return f"ClientConfig({self.to_dict()})"

    def to_dict(self):
        return {
            "search_worker_addrs": self.search_worker_addrs,
            "preferred_search_worker": self.preferred_search_worker,
            "hakes_addr": self.hakes_addr,
            "embed_endpoint_type": self.embed_endpoint_type,
            "embed_endpoint_config": self.embed_endpoint_config,
            "store_type": self.store_type,
            "store_addr": self.store_addr,
        }
