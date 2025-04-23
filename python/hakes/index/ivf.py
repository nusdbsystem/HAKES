import copy
import io
import logging
import struct
import torch
import numpy as np
from sklearn.cluster import KMeans


from .debug import centroids_to_str


class HakesIVF(torch.nn.Module):
    def __init__(
        self,
        d: int,
        nlist: int,
        centroids: np.ndarray,
        metric: str,
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
        self.metric = metric

        logging.info(
            f"Initialized IVF with d: {d}, nlist: {nlist}, centroids: {centroids.shape} metric: {metric}"
        )

    def __str__(self):
        return (
            super().__str__()
            + f" d: {self.d}, nlist: {self.nlist} metric: {self.metric}\n"
            + centroids_to_str(self.centroids)
        )

    @classmethod
    def from_reader(cls, reader: io.BufferedReader):
        d = struct.unpack("<i", reader.read(4))[0]
        _ = struct.unpack("<Q", reader.read(8))[0]
        metric_type = struct.unpack("B", reader.read(1))[0]
        if metric_type == 0:
            metric = "l2"
        else:
            metric = "ip"
        nlist = struct.unpack("<i", reader.read(4))[0]
        centroids = np.frombuffer(reader.read(nlist * d * 4), dtype=np.float32).reshape(
            nlist, d
        )
        return cls(d, nlist, centroids, metric)

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
        if self.metric == "ip":
            ip_scores = torch.matmul(vecs, self.centroids.T)
            return torch.argmax(ip_scores, dim=-1)
        else:
            norms = torch.norm(vecs[:, None, :] - self.centroids, dim=-1)
            return torch.argmin(norms, dim=1)

    def select_centers(self, vecs: torch.Tensor):
        if self.metric == "ip":
            scores = torch.matmul(vecs, self.centroids.T)
            assign = torch.argmax(scores, dim=-1)
            return self.centroids[assign]
        else:
            norms = torch.norm(vecs[:, None, :] - self.centroids, dim=-1)
            assign = torch.argmin(norms, dim=1)
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

    def save_to_writer(self, writer: io.BufferedWriter):
        """
        format (little endian)
        use_residual: int32 (0 no 1 yes)
        nlist: int32
        d: int32
        centroids: float32 array
        """

        logging.info(f"Saving IVF")
        writer.write(struct.pack("<i", self.d))
        writer.write(struct.pack("<Q", 0))
        writer.write(struct.pack("B", 0 if self.metric == "l2" else 1))
        writer.write(struct.pack("<i", self.nlist))
        writer.write(
            np.ascontiguousarray(
                self.centroids.detach().cpu().numpy(), dtype="<f"
            ).tobytes()
        )


def kmeans_ivf(data: np.ndarray, nlist: int, metric, niter: int = 30):
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
    print(f"iteration: {kmeans.n_iter_}")
    centroids = kmeans.cluster_centers_
    return HakesIVF(d, nlist, centroids, metric)
