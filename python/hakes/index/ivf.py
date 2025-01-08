import copy
import logging
import os
import struct
import torch
import numpy as np
from sklearn.cluster import KMeans


from typing import Dict

from .debug import centroids_to_str


class HakesIVF(torch.nn.Module):
    def __init__(
        self,
        d: int,
        nlist: int,
        centroids: np.ndarray,
        by_residual: bool = False,
    ):
        super().__init__()
        assert centroids.shape == (nlist, d)

        self.d = d
        self.nlist = nlist
        self.centroids = torch.nn.Parameter(
            torch.FloatTensor(copy.deepcopy(centroids)), requires_grad=True
        )
        self.num_centroids = len(centroids)
        self.by_residual = by_residual

        logging.info(
            f"Initialized IVF with d: {d}, nlist: {nlist}, centroids: {centroids.shape}"
        )

    def __str__(self):
        return (
            super().__str__()
            + f" d: {self.d}, nlist: {self.nlist}\n"
            + centroids_to_str(self.centroids)
        )

    @classmethod
    def from_bin_file(cls, ivf_file: str):
        with open(ivf_file, "rb") as f:
            by_residual = struct.unpack("<i", f.read(4))[0] == 1
            nlist = struct.unpack("<i", f.read(4))[0]
            d = struct.unpack("<i", f.read(4))[0]
            centroids = np.frombuffer(f.read(), dtype=np.float32).reshape(nlist, d)
            return cls(d, nlist, centroids, by_residual)

    def reduce_dim(self, target_d):
        if target_d > self.d:
            raise ValueError(
                f"target_d {target_d} must be less than config.d {self.config.d}"
            )

        self.d = target_d
        self.centroids.requires_grad = False
        self.centroids = torch.nn.Parameter(
            self.centroids[:, :target_d], requires_grad=True
        )
        return True

    def get_assignment(self, vecs: torch.Tensor):
        ip_scores = torch.matmul(vecs, self.centroids.T)
        return torch.argmax(ip_scores, dim=-1)

    def select_centers(self, vecs: torch.Tensor):
        ip_scores = torch.matmul(vecs, self.centroids.T)
        assign = torch.argmax(ip_scores, dim=-1)
        return self.centroids[assign]

    def update_centers(self, new_centers):
        self.centroids.data = torch.FloatTensor(new_centers)
        self.d = new_centers.shape[1]
        self.nlist = new_centers.shape[0]
        self.num_centroids = len(new_centers)
        print(f"new centers: {new_centers.shape}")

    def normalize_centers(self):
        self.centroids.data = torch.nn.functional.normalize(
            self.centroids.data, p=2, dim=-1
        )

    def save_as_bin(self, save_path, file_name="ivf.bin"):
        """
        format (little endian)
        use_residual: int32 (0 no 1 yes)
        nlist: int32
        d: int32
        centroids: float32 array
        """

        logging.info(f"Saving IVF to {save_path}")
        os.makedirs(save_path, exist_ok=True)
        with open(os.path.join(save_path, file_name), "wb") as f:
            f.write(struct.pack("<i", 1 if self.by_residual else 0))
            f.write(struct.pack("<i", self.centroids.shape[0]))
            f.write(struct.pack("<i", self.centroids.shape[1]))
            f.write(
                np.ascontiguousarray(
                    self.centroids.detach().cpu().numpy(), dtype="<f"
                ).tobytes()
            )

def kmeans_ivf (data: np.ndarray, nlist: int, niter: int = 300):
    """
    K-means clustering for IVF initialization

    Args:
      data: numpy array of shape (N, d)
      nlist: number of clusters
      niter: number of iterations
    """
    _, d = data.shape
    kmeans = KMeans(n_clusters=nlist, max_iter=niter)
    kmeans.fit(data)
    centroids = kmeans.cluster_centers_
    return HakesIVF(d, nlist, centroids)
