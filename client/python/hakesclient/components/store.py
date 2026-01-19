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

from typing import List, Tuple


# interface for extension to use external storage services
class Store:
    def __init__(self, addr: str):
        """
        Initialize the Store with the address of the storage service.

        Args:
            addr (str): The address of the storage service.
        """
        self.addr = addr

    def put(
        self, keys: List[str], value: List[bytes], xids: List[bytes] | None
    ) -> Tuple[bool, List[str]]:
        """
        Put a key-value pair into the store.

        Args:
            keys (List[str]): A list of keys to associate with the value.
            value (List[bytes]): The value to store, which can be a list of bytes.
            xids (List[bytes] | None): Optional list of fixed-length external IDs to associate with the keys.

        Returns:
            Tuple[bool, List[str]]: A tuple containing a boolean indicating success or failure,
                                    and an optional list of fixed-length uids that were successfully stored.
        """
        raise NotImplementedError("put method must be implemented by subclass")

    def get_by_keys(self, keys: List[str]) -> List[bytes]:
        """
        Get values by a list of keys from the store.

        Args:
            keys (List[str]): A list of keys to retrieve.

        Returns:
            bytes: The value associated with the key.
        """
        raise NotImplementedError("get method must be implemented by subclass")

    def get_by_ids(self, xids: List[bytes]) -> list[bytes]:
        """
        Get values by a list of keys from the store.

        Args:
            keys (list[str]): A list of keys to retrieve.

        Returns:
            list[bytes]: A list of values associated with the keys.
        """
        raise NotImplementedError("get_by_ids method must be implemented by subclass")

    def delete(self, key: str) -> bool:
        """
        Delete a key-value pair from the store.

        Args:
            key (str): The key to delete.

        Returns:
            bool: True if the operation was successful, False otherwise.
        """
        raise NotImplementedError("delete method must be implemented by subclass")
