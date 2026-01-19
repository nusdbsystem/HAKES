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

import numpy as np
from typing import List

# embedder interface for extension to use external embedding services
class Embedder:
    def __init__(self):
        pass

    def embed_text(self, texts: List[str]) -> np.ndarray:
        """
        Embed the given text using the embedding service.

        Args:
            text (List[str]): A list of text strings to be embedded.

        Returns:
            np.ndarray: An array of embedded vectors corresponding to the input text.
        """
        raise NotImplementedError("embed_text method must be implemented by subclass")
    
    def embed_binary(self, binary_data: List[bytes]) -> np.ndarray:
        """
        Embed the given binary data using the embedding service.

        Args:
            binary_data (List[bytes]): A list of binary data to be embedded.

        Returns:
            np.ndarray: An array of embedded vectors corresponding to the input binary data.
        """
        raise NotImplementedError("embed_binary method must be implemented by subclass")
