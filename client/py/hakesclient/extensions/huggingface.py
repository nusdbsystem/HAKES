from typing import List, Optional
from huggingface_hub.inference._providers import PROVIDER_OR_POLICY_T
import numpy as np
import os
from hakesclient.components.embedder import Embedder
from huggingface_hub import InferenceClient


class HuggingFaceEmbedder(Embedder):
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        provider: Optional[PROVIDER_OR_POLICY_T] = "auto",
    ):
        super().__init__()
        self.api_key = api_key or os.getenv("HF_API_KEY")
        # Only use endpoint_url argument or HF_ENDPOINT_URL env var
        self.client = InferenceClient(
            provider=provider,
            model=model,
            token=self.api_key,
        )
        if not self.api_key:
            raise ValueError(
                "HuggingFace API key must be provided via argument or HF_API_KEY env var."
            )

    @classmethod
    def from_config(cls, config: dict):
        if config.get("model") is None:
            raise ValueError("Model is required")

        return cls(
            api_key=config.get("api_key"),
            model=config.get("model"),
            provider=config.get("provider", "auto"),
        )

    def embed_text(self, texts: List[str]) -> np.ndarray:
        all_embeddings = []
        for i in range(0, len(texts)):
            response = self.client.feature_extraction(texts[i])
            # The response is expected to be a list of embeddings
            all_embeddings.append(response.data)
        return np.array(all_embeddings, dtype=np.float32)

    def embed_binary(self, binary_data: List[bytes]) -> np.ndarray:
        raise NotImplementedError(
            "embed_binary is not supported for HuggingFaceEmbedder."
        )
