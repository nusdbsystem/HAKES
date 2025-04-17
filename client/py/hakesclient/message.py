"""
This file contains the message format for the client and server to communicate.
ref: HAKES/message
"""

import numpy as np
from typing import Dict


def encode_search_worker_load_collection_request(
    collection_name: str,
    user_id: str = "",
    ks_addr: str = "",
    ks_port: int = -1,
):
    data = {
        "collection_name": collection_name,
        "user_id": user_id,
        "ks_addr": ks_addr,
        "ks_port": ks_port,
    }
    return data


def decode_search_worker_load_collection_response(resp: Dict) -> Dict:
    return resp


def encode_search_worker_add_request(
    collection_name: str,
    vecs: np.ndarray,
    ids: np.ndarray,
    user_id: str = "",
    ks_addr: str = "",
    ks_port: int = -1,
):
    data = {
        "collection_name": collection_name,
        "d": vecs.shape[1],
        "vecs": np.ascontiguousarray(vecs, dtype="<f").tobytes().hex(),
        "ids": np.ascontiguousarray(ids, dtype="<q").tobytes().hex(),
        "user_id": user_id,
        "ks_addr": ks_addr,
        "ks_port": ks_port,
    }
    return data


def decode_search_worker_add_response(resp: Dict) -> Dict:
    return resp


def encode_search_worker_search_request(
    collection_name: str,
    k: int,
    vecs: np.ndarray,
    nprobe: int,
    k_factor: int,
    metric_type: str,  # L2: 0, IP: 1
    user_id: str = "",
    ks_addr: str = "",
    ks_port: int = -1,
):
    # flatten the vecs
    d = vecs.shape[1]
    vecs = vecs.flatten()
    data = {
        "collection_name": collection_name,
        "d": d,
        "k": k,
        "vecs": np.ascontiguousarray(vecs, dtype="<f").tobytes().hex(),
        "nprobe": nprobe,
        "k_factor": k_factor,
        "metric_type": 0 if metric_type == "L2" else 1,
        "user_id": user_id,
        "ks_addr": ks_addr,
        "ks_port": ks_port,
    }
    return data


def decode_search_worker_search_response(
    resp: Dict, k: int, filter_invalid: bool = True
) -> Dict:
    if "status" not in resp or not resp["status"]:
        return resp
    # print("parsing response")
    ids = (
        np.frombuffer(bytes.fromhex(resp["ids"]), dtype=np.int64)
        .reshape(-1, k)
        .tolist()
    )
    scores =(
        np.frombuffer(bytes.fromhex(resp["scores"]), dtype=np.float32)
        .reshape(-1, k)
        .tolist()
    )
    for i in range(len(ids)):
        if filter_invalid:
            ids[i] = ids[i][ids[i] != -1]
    resp["ids"] = ids
    resp["scores"] = scores
    return resp


def encode_search_worker_rerank_request(
    collection_name: str,
    k: int,
    vecs: np.ndarray,
    input_ids: np.ndarray,
    metric_type: str,  # L2: 0, IP: 1
    user_id: str = "",
    ks_addr: str = "",
    ks_port: int = -1,
):
    # flatten the vecs
    if metric_type not in ["L2", "IP"]:
        raise ValueError("metric_type must be one of ['L2', 'IP']")
    if input_ids.shape[0] != vecs.shape[0]:
        raise ValueError("input_ids shape does not match vecs")
    d = vecs.shape[1]
    vecs = vecs.flatten()
    data = {
        "collection_name": collection_name,
        "d": d,
        "k": k,
        "vecs": np.ascontiguousarray(vecs, dtype="<f").tobytes().hex(),
        "input_ids": np.ascontiguousarray(input_ids, dtype="<q").tobytes().hex(),
        "metric_type": 0 if metric_type == "L2" else 1,
        "user_id": user_id,
        "ks_addr": ks_addr,
        "ks_port": ks_port,
    }
    return data


def decode_search_worker_rerank_response(resp: Dict, k: int) -> Dict:
    if "status" not in resp or not resp["status"]:
        return resp
    resp["ids"] = (
        np.frombuffer(bytes.fromhex(resp["ids"]), dtype=np.int64)
        .reshape(-1, k)
        .tolist()
    )
    resp["scores"] = (
        np.frombuffer(bytes.fromhex(resp["scores"]), dtype=np.float32)
        .reshape(-1, k)
        .tolist()
    )
    return resp

def encode_search_worker_delete_request(collection_name: str, ids: np.ndarray):
    data = {
        "collection_name": collection_name,
        "ids": np.ascontiguousarray(ids, dtype="<q").tobytes().hex(),
    }
    return data


def decode_search_worker_delete_response(resp: Dict) -> Dict:
    return {"status": resp["status"], "msg": resp["msg"]}
