import json
import os
from typing import List


class ClientConfig:
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
                data["embed_endpoint_addr"],
                data["store_type"],
                data["store_addr"],
            )
        except KeyError as e:
            raise ValueError(f"Missing field {e} in {path}")

        return conf

    def __init__(
        self,
        search_worker_addrs: List[str] = [],
        preferred_search_worker: int = 0,
        hakes_addr: str = "",
        embed_endpoint_type: str = "",
        embed_endpoint_config: str = "",
        embed_endpoint_addr: str = "",
        store_type: str = "",
        store_addr: str = "",
    ):
        self.search_worker_addrs = search_worker_addrs
        self.preferred_search_worker = preferred_search_worker
        self.hakes_addr = hakes_addr
        self.embed_endpoint_type = (
            ""
            if (not embed_endpoint_type or embed_endpoint_type == "none")
            else embed_endpoint_type
        )
        self.embed_endpoint_config = embed_endpoint_config
        self.embed_endpoint_addr = (
            ""
            if not self.embed_endpoint_type or not embed_endpoint_addr
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

        self.n = len(search_worker_addrs)

    def save(self, path):
        # save as a json file
        json.dump(self.__dict__, path)

    def __repr__(self) -> str:
        return f"ClientConfig({self.to_dict()})"

    def to_dict(self):
        return {
            "search_worker_addrs": self.search_worker_addrs,
            "preferred_search_worker": self.preferred_search_worker,
            "hakes_addr": self.hakes_addr,
            "embed_endpoint_type": self.embed_endpoint_type,
            "embed_endpoint_config": self.embed_endpoint_config,
            "embed_endpoint_addr": self.embed_endpoint_addr,
            "store_type": self.store_type,
            "store_addr": self.store_addr,
        }
