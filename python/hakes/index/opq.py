import numpy as np
from sklearn.cluster import KMeans

from .pq import HakesPQ
from .vt import HakesVecTransform


def opq_train(data: np.ndarray, opq_out: int, m: int, iter: int = 20):
    if opq_out > data.shape[1]:
        raise ValueError(
            "opq_out must be less than or equal to the dimension of the data"
        )
    if opq_out % m != 0:
        raise ValueError("opq_out must be divisible by m")

    d = data.shape[1]
    data_t = data.T

    rand_A = np.random.randn(d, d)
    # qr decomposition
    Q, _ = np.linalg.qr(rand_A)
    # check if Q is orthonormal
    assert np.allclose(np.eye(d), Q @ Q.T)
    opq_vt_A = Q[:, :opq_out]
    pq_codebook = np.random.randn(m, 2**4, opq_out // m)

    for _ in range(iter):
        projected_data = data @ opq_vt_A
        for i in range(m):
            kmeans = KMeans(n_clusters=2**4)
            data_sub = projected_data[:, i * (opq_out // m) : (i + 1) * (opq_out // m)]
            kmeans.fit(data_sub)
            pq_codebook[i] = kmeans.cluster_centers_
        reconstructed_data = np.zeros_like(projected_data)
        for i in range(m):
            assignments = np.argmin(
                np.linalg.norm(
                    projected_data[:, i * (opq_out // m) : (i + 1) * (opq_out // m)][
                        :, None
                    ]
                    - pq_codebook[i],
                    axis=2,
                ),
                axis=1,
            )
            reconstructed_data[:, i * (opq_out // m) : (i + 1) * (opq_out // m)] = (
                pq_codebook[i][assignments]
            )
        print(f"recon error: {np.linalg.norm(projected_data - reconstructed_data)}")
        U, _, V = np.linalg.svd(data_t @ reconstructed_data)
        opq_vt_A = U[:opq_out] @ V

    # print(opq_vt_A)
    return HakesVecTransform(d, opq_out, opq_vt_A.T, np.zeros(opq_out)), HakesPQ(
        opq_out, m, nbits=4, codebook=pq_codebook, fixed_assignment=True
    )
