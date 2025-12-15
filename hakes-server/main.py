import os
import sys
import time
import json
import uuid
import hashlib
from typing import Optional, List

from fastapi import FastAPI, HTTPException, Header, Depends
from pydantic import BaseModel

# Ensure client package is importable (client/py is a sibling directory)
ROOT = os.path.dirname(os.path.dirname(__file__))
CLIENT_PY = os.path.abspath(os.path.join(ROOT, "client", "py"))
if CLIENT_PY not in sys.path:
    sys.path.insert(0, CLIENT_PY)

from hakesclient.extensions.mongodb import MongoDB
from hakesclient.components.searcher import Searcher
from hakesclient.extensions.huggingface import HuggingFaceEmbedder


def hash_password(password: str, salt: Optional[bytes] = None) -> dict:
    if salt is None:
        salt = os.urandom(16)
    digest = hashlib.sha256(salt + password.encode("utf-8")).hexdigest()
    return {"salt": salt.hex(), "hash": digest}


def verify_password(stored: dict, password: str) -> bool:
    salt = bytes.fromhex(stored.get("salt", ""))
    expected = stored.get("hash", "")
    digest = hashlib.sha256(salt + password.encode("utf-8")).hexdigest()
    return digest == expected


class RegisterReq(BaseModel):
    username: str
    password: str


class LoginReq(BaseModel):
    username: str
    password: str


class LoadReq(BaseModel):
    collection_name: str


class AddReq(BaseModel):
    collection_name: str
    keys: List[str]
    values: List[str]
    data_type: str
    ids: Optional[List[str]] = None


class SearchReq(BaseModel):
    collection_name: str
    query: str
    data_type: str
    k: int
    nprobe: int
    k_factor: int = 1
    metric_type: str = "IP"


app = FastAPI(title="HAKES Server")

# Configuration from environment
STORE_ADDR = os.getenv("HAKES_STORE_ADDR", "mongodb://localhost:27017")
SEARCHER_ADDR = os.getenv("HAKES_SEARCHER_ADDR", "http://localhost:8001")
EMBEDDER_TYPE = os.getenv("HAKES_EMBEDDER", "huggingface")
HF_MODEL = os.getenv("HF_MODEL", "all-MiniLM-L6-v2")
HF_API_KEY = os.getenv("HF_API_KEY", None)

# instantiate components
store = MongoDB(STORE_ADDR, db_name="hakes_server", collection_name="kv")
searcher = Searcher([SEARCHER_ADDR])
embedder = None
if EMBEDDER_TYPE == "huggingface":
    embedder = HuggingFaceEmbedder(api_key=HF_API_KEY, model=HF_MODEL)


def _get_token_from_header(authorization: Optional[str]) -> str:
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(status_code=401, detail="Invalid Authorization header")
    return parts[1]


def validate_token(authorization: Optional[str] = Header(None)) -> str:
    token = _get_token_from_header(authorization)
    key = f"token:{token}"
    values, _ = store.get_by_keys([key])
    if not values or values[0] == b"":
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    try:
        data = json.loads(values[0])
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid token data")
    expiry = data.get("expiry", 0)
    if expiry != 0 and time.time() > expiry:
        raise HTTPException(status_code=401, detail="Token expired")
    return data.get("username")


@app.post("/register")
def register(req: RegisterReq):
    key = f"user:{req.username}"
    vals, _ = store.get_by_keys([key])
    if vals and vals[0] != b"":
        raise HTTPException(status_code=400, detail="User already exists")
    pw = hash_password(req.password)
    user_obj = {"username": req.username, "salt": pw["salt"], "hash": pw["hash"]}
    ok, _ = store.put([key], [json.dumps(user_obj).encode("utf-8")], None)
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to create user")
    return {"ok": True}


@app.post("/login")
def login(req: LoginReq):
    key = f"user:{req.username}"
    vals, _ = store.get_by_keys([key])
    if not vals or vals[0] == b"":
        raise HTTPException(status_code=401, detail="Invalid username or password")
    try:
        user = json.loads(vals[0])
    except Exception:
        raise HTTPException(status_code=500, detail="Corrupted user data")
    if not verify_password(user, req.password):
        raise HTTPException(status_code=401, detail="Invalid username or password")
    token = uuid.uuid4().hex
    expiry = int(time.time()) + 7 * 24 * 3600
    token_obj = {"username": req.username, "expiry": expiry}
    ok, _ = store.put([f"token:{token}"], [json.dumps(token_obj).encode("utf-8")], None)
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to create token")
    return {"access_token": token, "expiry": expiry}


@app.post("/load_collection")
def load_collection(req: LoadReq, username: str = Depends(validate_token)):
    res = searcher.load_collection(req.collection_name)
    if res is None:
        raise HTTPException(status_code=500, detail="searcher failed to load collection")
    return res


@app.post("/add")
def add(req: AddReq, username: str = Depends(validate_token)):
    if embedder is None:
        raise HTTPException(status_code=500, detail="No embedder configured")
    if req.data_type == "text":
        vectors = embedder.embed_text(req.values)
    else:
        binary_blobs = [v.encode("utf-8") for v in req.values]
        vectors = embedder.embed_binary(binary_blobs)

    raw_values = [v.encode("utf-8") for v in req.values]
    ok, xids = store.put(req.keys, raw_values, None)
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to store data")

    res = searcher.add(req.collection_name, vectors, None if req.ids is None else req.ids)
    if res is None:
        raise HTTPException(status_code=500, detail="searcher add failed")
    return {"ok": True, "searcher": res}


@app.post("/search")
def search(req: SearchReq, username: str = Depends(validate_token)):
    if embedder is None:
        raise HTTPException(status_code=500, detail="No embedder configured")
    if req.data_type == "text":
        qvec = embedder.embed_text([req.query])
    else:
        qvec = embedder.embed_binary([req.query.encode("utf-8")])

    res = searcher.search(req.collection_name, qvec, req.k, req.nprobe, req.k_factor, req.metric_type)
    if res is None:
        raise HTTPException(status_code=500, detail="searcher search failed")

    ids = res.get("ids", [])
    if isinstance(ids, list) and len(ids) > 0 and isinstance(ids[0], list):
        ids_to_fetch = ids[0]
    else:
        ids_to_fetch = ids
    values = store.get_by_ids(ids_to_fetch)
    return {"searcher": res, "values": values}


@app.post("/checkpoint")
def checkpoint(req: LoadReq, username: str = Depends(validate_token)):
    res = searcher.checkpoint(req.collection_name)
    if res is None:
        raise HTTPException(status_code=500, detail="searcher checkpoint failed")
    return res


@app.post("/delete")
def delete(req: LoadReq, username: str = Depends(validate_token)):
    # For now rely on searcher/store clients to perform delete behavior
    return {"ok": True}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("HAKES_SERVER_PORT", 8000)))
