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


def validate_addr(addr: str) -> str:
    """
    Validate and format the address.

    Args:
        addr (str): The address to validate.

    Returns:
        str: The validated and formatted address.

    Raises:
        ValueError: If the address is empty.
    """
    if not addr:
        raise ValueError("address is empty")

    if not addr.startswith("http"):
        addr = "http://" + addr

    return addr

def ids_to_xids(ids):
    """
    Convert a list of integer IDs to byte array XIDs.

    Args:
        ids (list[int]): A list of integer IDs.

    Returns:
        list[bytes]: A list of byte array XIDs.
    """
    xids = [id.to_bytes(8, 'big', signed=True) for id in ids]
    return xids

def xids_to_ids(xids):
    """
    Convert a list of byte array XIDs to integer IDs.

    Args:
        xids (list[bytes]): A list of byte array XIDs.

    Returns:
        list[int]: A list of integer IDs.
    """
    ids = [int.from_bytes(xid, 'big', signed=True) for xid in xids]
    return ids

def texts_to_bytes(texts):
    """
    Convert a list of text strings to byte arrays.

    Args:
        texts (list[str]): A list of text strings.

    Returns:
        list[bytes]: A list of byte arrays.
    """
    return [text.encode('utf-8') for text in texts]

def bytes_to_texts(bytes):
    """
    Convert a list of byte arrays to text strings.

    Args:
        byte_arrays (list[bytes]): A list of byte arrays.

    Returns:
        list[str]: A list of text strings.
    """
    return [b.decode('utf-8') for b in bytes]
