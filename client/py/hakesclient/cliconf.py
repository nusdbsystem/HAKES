import json
import os
from typing import List


class ClientConfig:
    def __init__(
        self,
        path: str = None,
        search_worker_addrs: List[str] = [],
        preferred_search_worker: int = 0,
        hakes_addr: str = "",
        embed_endpoint_type: str = "",
        embed_endpoint_config: str = "",
        embed_endpoint_addr: str = "",
        store_type: str = "",
        store_addr: str = "",
    ):
        # by default, load from a json file
        if not path:
            # load the default path if not specified: $HOME/.hakes/client_config.json
            path = os.path.join(os.path.expanduser("~"), ".hakes", "config.json")

        # load from a json file
        with open(path, "r") as f:
            data = json.load(f)
            if search_worker_addrs == None or search_worker_addrs == []:
                search_worker_addrs = data["search_worker_addrs"]
            if preferred_search_worker == -1:
                preferred_search_worker = data["preferred_search_worker"]
            if hakes_addr == "" or hakes_addr == None:
                hakes_addr = data["hakes_addr"]
            if embed_endpoint_type == "" or embed_endpoint_type == None:
                embed_endpoint_type = data["embed_endpoint_type"]
            if embed_endpoint_config == "" or embed_endpoint_config == None:
                embed_endpoint_config = data["embed_endpoint_config"]
            if embed_endpoint_addr == "" or embed_endpoint_addr == None:
                embed_endpoint_addr = data["embed_endpoint_addr"]
            if store_type == "" or store_type == None:
                store_type = data["store_type"]
            if store_addr == "" or store_addr == None:
                store_addr = data["store_addr"]

        self.search_worker_addrs = search_worker_addrs
        self.n = len(search_worker_addrs)
        self.preferred_search_worker = preferred_search_worker
        self.hakes_addr = hakes_addr
        self.embed_endpoint_type = (
            ""
            if embed_endpoint_type == "none"
            or embed_endpoint_type == None
            or embed_endpoint_addr == ""
            else embed_endpoint_type
        )
        self.embed_endpoint_config = embed_endpoint_config
        self.embed_endpoint_addr = (
            ""
            if embed_endpoint_type == "none" or embed_endpoint_addr == None
            else embed_endpoint_addr
        )
        self.store_type = (
            ""
            if store_type == "none" or store_type == None or store_addr == ""
            else store_type
        )
        self.store_addr = (
            "" if store_type == "none" or store_addr == None else store_addr
        )
        print(self)

    def __repr__(self) -> str:
        return f"ClientConfig: search_worker_addrs: {self.search_worker_addrs}, preferred_search_worker: {self.preferred_search_worker}, hakes_addr: {self.hakes_addr}, embed_endpoint_type: {self.embed_endpoint_type}, embed_endpoint_config: {self.embed_endpoint_config}, embed_endpoint_addr: {self.embed_endpoint_addr}, store_type: {self.store_type}, store_addr: {self.store_addr}"

    def save(self, path):
        # save as a json file
        json.dump(self.__dict__, path)
