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
