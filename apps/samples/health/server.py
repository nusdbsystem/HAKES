import os
import sys
from dotenv import load_dotenv

load_dotenv("/run/secrets/env_secret")
sys.path.append("../..")

from hakesclient import Client, DataType
from hakesclient.utils import texts_to_bytes
from hakesclient.utils import bytes_to_texts
from common.llm.healthgpt import HealthGPT
from fastapi import FastAPI

GITEEAI_KEY = os.getenv("GITEEAI_KEY")
COLLECTION_NAME = "pubmedqa"

app = FastAPI(
    title="Health Sample App",
    description="A sample application for using HAKES with HealthGPT",
    version="0.1.0",
)


class Service:
    def __init__(self, embedder_url: str, searcher_url: str, store_url: str):
        from hakesclient.components.store import Store
        from hakesclient.components.embedder import Embedder
        from hakesclient.components.searcher import Searcher
        from hakesclient.extensions.mongodb import MongoDB
        from hakesclient.extensions.ollama import OllamaEmbedder

        self.store: Store = MongoDB(store_url, "collections", "pubmedqa")
        self.embedder: Embedder = OllamaEmbedder(
            base_url=embedder_url, model="nomic-embed-text"
        )
        self.searcher: Searcher = Searcher([searcher_url])
        self.client: Client = Client(
            embedder=self.embedder,
            store=self.store,
            searcher=self.searcher,
        )
        self.client.load_collection(COLLECTION_NAME)
        self.llm = HealthGPT(
            model_name="HealthGPT-L14",
            config={"api_key": GITEEAI_KEY, "base_url": "https://ai.gitee.com/v1"},
        )


service = None


@app.get("/datasets/list")
def list_datasets():
    pass


@app.post("/chat")
def chat_endpoint(data: dict):
    if data.get("question") is None:
        return {"error": "question is required"}
    question = data["question"]
    search_result = bytes_to_texts(
        service.client.search(
            COLLECTION_NAME, texts_to_bytes([question]), DataType.TEXT, 3, 100, 5, "IP"
        )
    )
    context_info = ""
    for i, d in enumerate(search_result):
        if not d:
            continue
        context_info += f"{i + 1}：{d}\n"
    prompt = f"""
    Please answer the following question based only on the context provided.\n
    <Context>\n
    {context_info}
    </Context>\n
    Question: {question}\n
    Answer:
    """
    response = service.llm.generate_text(prompt, max_length=300)
    return {"answer": response}


@app.get("/health")
def health_check():
    return {"status": "ok"}


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="Health Sample App Server")
    parser.add_argument(
        "--embedder_url",
        type=str,
        default="http://localhost:11434",
        help="Address of the embedder",
    )
    parser.add_argument(
        "--searcher_url",
        type=str,
        default="http://localhost:2053",
        help="Address of the search worker",
    )
    parser.add_argument(
        "--store_url",
        type=str,
        default="mongodb://localhost:27017",
        help="Path to the MongoDB database",
    )
    args = parser.parse_args()
    # print each argument
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    return args


if __name__ == "__main__":
    import uvicorn

    args = parse_args()
    service = Service(args.embedder_url, args.searcher_url, args.store_url)

    uvicorn.run(app, host="0.0.0.0", port=2033)

# curl -X POST http://localhost:2033/chat -H "Content-Type: application/json" -d '{"question": "What are group 2 innate lymphoid cells?"}'
