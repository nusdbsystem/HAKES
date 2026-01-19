import numpy as np
import os
from openai import OpenAI
from typing import List, Optional
from hakesclient.components.embedder import Embedder


DEFAULT_MODEL = "text-embedding-3-small"


class OpenAIEmbedder(Embedder):
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
    ):
        super().__init__()
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key must be provided via argument or OPENAI_API_KEY env var."
            )
        self.model = model
        self.client = OpenAI(api_key=self.api_key)

    @classmethod
    def from_config(cls, config: dict):
        return cls(
            api_key=config.get("api_key"),
            model=config.get("model", DEFAULT_MODEL),
        )

    def embed_text(self, texts: List[str]) -> np.ndarray:
        # OpenAI API supports batching, but has a max batch size (e.g., 2048 tokens or 2048 items for some models)
        # We'll use a reasonable batch size (e.g., 512)
        batch_size = 512
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            response = self.client.embeddings.create(input=batch, model=self.model)
            # response.data is a list of dicts with 'embedding' key
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)
        return np.array(all_embeddings, dtype=np.float32)

    def embed_binary(self, binary_data: List[bytes]) -> np.ndarray:
        raise NotImplementedError("embed_binary is not supported for OpenAIEmbedder.")
