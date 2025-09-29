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

from ..components.store import Store
from pymongo import MongoClient, UpdateOne
from typing import List, Tuple
import struct


class MongoDB(Store):
    """
    MongoDB store implementation for HAKES.
    This class extends the Store interface to provide MongoDB-specific functionality.
    """

    def _validate_mongodb_address(self, addr: str) -> str:
        """
        Validate and format the MongoDB address.

        Args:
            addr (str): The address to validate.

        Returns:
            str: The validated and formatted MongoDB address.

        Raises:
            ValueError: If the address is empty or invalid.
        """
        if not addr:
            raise ValueError("MongoDB address is empty")

        if not addr.startswith("mongodb://"):
            addr = "mongodb://" + addr

        return addr

    def __init__(
        self, addr: str, db_name: str = "hakes", collection_name: str = "default"
    ):
        """
        Initialize the MongoDB store with the address of the MongoDB server.

        Args:
            addr (str): The address of the MongoDB server.
            db_name (str): The database name to use.
            collection_name (str): The collection name to use.
        """
        super().__init__(addr)
        self.addr = self._validate_mongodb_address(addr)
        self.client = MongoClient(self.addr)
        self.db = self.client[db_name]
        self.collection = self.db[collection_name]
        # Ensure index on key for fast lookup
        self.collection.create_index("key", unique=True)
        # Counter collection for xid auto-increment
        self.counters = self.db["counters"]
        # Initialize counter if not present
        if self.counters.find_one({"_id": collection_name}) is None:
            self.counters.insert_one({"_id": collection_name, "seq": 0})
        self._counter_key = collection_name

    def connected(self) -> bool:
        """
        Check if the MongoDB connection is established.
        """
        return self.client.admin.command("ismaster")["ok"] == 1

    def connect(self):
        """
        Connect to the MongoDB server (no-op, handled by MongoClient).
        """
        pass

    def disconnect(self):
        """
        Disconnect from the MongoDB server.
        """
        if hasattr(self, "client"):
            self.client.close()

    # deletor to call disconnect
    def __del__(self):
        """
        Destructor to ensure the MongoDB connection is closed when the object is deleted.
        """
        try:
            self.disconnect()
        except Exception:
            pass

    def _get_next_xids(self, batch_size: int) -> list[bytes]:
        # Atomically increment the counter by batch_size and get the starting value
        doc = self.counters.find_one_and_update(
            {"_id": self._counter_key},
            {"$inc": {"seq": batch_size}},
            return_document=True,
            upsert=True,
        )
        # The new value of seq is the last reserved id, so calculate the range
        last_id = doc["seq"]
        first_id = last_id - batch_size + 1
        # always encode to 8 bytes, big-endian signed
        return [struct.pack(">q", i) for i in range(first_id, last_id + 1)]

    def put(
        self, keys: List[str], value: List[bytes], xids: List[bytes] | None
    ) -> Tuple[bool, List[bytes]]:
        """
        Put key-value pairs into the store. Auto-increment xid if not provided, and batch the operation.
        XIDs are always stored as 8-byte (64-bit signed) values.
        """
        if len(keys) != len(value):
            raise ValueError("keys and value must have the same length")
        if xids and len(xids) != len(keys):
            raise ValueError("xids and keys must have the same length")

        try:
            n = len(keys)
            # Determine which indices need auto xids
            if xids is None:
                xids = self._get_next_xids(n)
            docs = []

            for i, key in enumerate(keys):
                doc = {"key": key, "value": value[i], "xid": xids[i]}
                docs.append(doc)
            # Prepare bulk upsert operations
            operations = [
                UpdateOne({"key": doc["key"]}, {"$set": doc}, upsert=True)
                for doc in docs
            ]
            self.collection.bulk_write(operations, ordered=False)
            return True, xids
        except Exception as e:
            return False, []

    def get_by_keys(self, keys: List[str]) -> Tuple[List[bytes], List[bytes]]:
        """
        Get values and xids by a list of keys from the store, using a batched query.
        Returns two lists: values and xids, in the same order as input keys.
        If a key is missing, returns b'' for value and b'' for xid.
        """
        if not keys:
            return [], []
        docs = self.collection.find({"key": {"$in": keys}})
        key_to_doc = {doc["key"]: doc for doc in docs if "key" in doc}
        values = [key_to_doc.get(key, {}).get("value", b"") for key in keys]
        xids = [key_to_doc.get(key, {}).get("xid", b"") for key in keys]
        return values, xids

    def get_by_ids(self, xids: List[bytes]) -> list[bytes]:
        """
        Get values by a list of xids from the store, using a batched query.
        Preserves the order of input xids in the output.
        """
        if not xids:
            return []
        # Convert xids to a list of unique values for the query
        xids_set = list(set(xids))
        # Query all at once
        docs = self.collection.find({"xid": {"$in": xids_set}})
        # Build a mapping from xid to value
        xid_to_value = {
            doc["xid"]: doc["value"] for doc in docs if "xid" in doc and "value" in doc
        }
        # Return results in the same order as input xids
        results = [xid_to_value.get(xid, b"") for xid in xids]
        return results

    def delete(self, key: str) -> bool:
        """
        Delete a key-value pair from the store.
        """
        result = self.collection.delete_one({"key": key})
        return result.deleted_count > 0


# https://www.mongodb.com/resources/products/platform/mongodb-auto-increment
# https://www.mongodb.com/resources/languages/python
