import logging
import numpy as np
import torch
from torch.utils.data import Dataset


def load_data_bin(file_path, N, d):
    data = np.zeros((N, d), dtype=np.float32)
    with open(file_path, "rb") as f:
        data = np.frombuffer(f.read(N * d * 4), dtype="<f").reshape(N, d)
    return data


def load_neighbors_bin(file_path, N, d):
    neighbors = np.zeros((N, d), dtype=np.int32)
    with open(file_path, "rb") as f:
        neighbors = np.frombuffer(f.read(N * d * 4), dtype="<i").reshape(N, d)
    return neighbors


class HakesDataset(Dataset):
    def __init__(
        self,
        train_data: np.ndarray,
        query: np.ndarray,
        pos_ids: np.ndarray,
    ):
        """
        Training dataset for Hakes index

        """
        assert train_data.shape[1] == query.shape[1]
        assert query.shape[0] == pos_ids.shape[0]
        self.train_data = train_data
        self.query = query
        self.pos_ids = pos_ids

    def __len__(self):
        return len(self.query)

    def __getitem__(self, idx):
        return (
            self.query[idx],
            self.train_data[self.pos_ids[idx]],
            self.pos_ids[idx],
        )

class HakesDataCollator:
    def __call__(self, batch):
        query_batch = np.array([x[0] for x in batch])
        pos_batch = np.array([x[1] for x in batch])
        pos_ids_batch = [x[2] for x in batch]
        query_tensor = torch.FloatTensor(query_batch)
        pos_tensor = torch.FloatTensor(pos_batch)
        return {
            "query_data": query_tensor,
            "pos_data": pos_tensor,
            "pos_ids": pos_ids_batch,
        }
