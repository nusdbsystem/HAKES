import numpy as np
from sklearn.cluster import KMeans
import tqdm


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

    for it in tqdm.trange(iter):
        projected_data = data @ opq_vt_A
        reconstructed_data = np.zeros_like(projected_data)
        for i in range(m):
            if it % 5 != 0:
                kmeans = KMeans(n_clusters=2**4, init=pq_codebook[i], max_iter=50)
            else:
                kmeans = KMeans(n_clusters=2**4, max_iter=20)
            data_sub = projected_data[:, i * (opq_out // m) : (i + 1) * (opq_out // m)]
            kmeans.fit(data_sub)
            pq_codebook[i] = kmeans.cluster_centers_

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
        # print(f"recon error: {np.linalg.norm(projected_data - reconstructed_data)}")
        U, _, V = np.linalg.svd(data_t @ reconstructed_data)
        opq_vt_A = U[:, :opq_out] @ V

    return opq_vt_A, pq_codebook
