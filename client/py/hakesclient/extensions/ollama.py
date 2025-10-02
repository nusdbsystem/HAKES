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
import os
import requests
from typing import List, Optional
from hakesclient.components.embedder import Embedder


class OllamaEmbedder(Embedder):
    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ):
        super().__init__()
        self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.model = model or os.getenv("OLLAMA_MODEL", "nomic-embed-text")
        
        # Ollama doesn't require API keys for local deployments
        self.headers = {"Content-Type": "application/json"}

    @classmethod
    def from_config(cls, config: dict):
        if config.get("model") is None:
            raise ValueError("Model is required")

        return cls(
            base_url=config.get("base_url"),
            model=config.get("model"),
        )

    def embed_text(self, texts: List[str]) -> np.ndarray:
        all_embeddings = []
        
        for text in texts:
            # Ollama embedding API endpoint (uses /api/embed, not /api/embeddings)
            url = f"{self.base_url}/api/embed"
            
            payload = {
                "model": self.model,
                "input": text  # Ollama uses 'input' not 'prompt' for embeddings
            }
            
            try:
                response = requests.post(url, json=payload, headers=self.headers)
                response.raise_for_status()
                
                # Extract embedding from response
                # Ollama returns "embeddings" format (batch)
                embedding_data = response.json()
                if "embeddings" in embedding_data:
                    embeddings = embedding_data["embeddings"]
                    all_embeddings.extend(embeddings)
                else:
                    raise ValueError(f"Expected 'embeddings' field in response, got: {embedding_data}")
                    
            except requests.exceptions.RequestException as e:
                raise RuntimeError(f"Failed to get embedding from Ollama: {e}")
            except (KeyError, ValueError) as e:
                raise RuntimeError(f"Failed to parse Ollama response: {e}")
        
        return np.array(all_embeddings, dtype=np.float32)

    def embed_binary(self, binary_data: List[bytes]) -> np.ndarray:
        raise NotImplementedError(
            "embed_binary is not supported for OllamaEmbedder."
        )
