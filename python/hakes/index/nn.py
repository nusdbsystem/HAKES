import numpy as np
import torch
import tqdm


def get_nn(
    data: np.ndarray,
    k: int,
    query: np.ndarray = None,
    sample_ratio: float = 0.01,
    seed: int = 0,
    device: str = "cpu",
    normalize: bool = True,
    distance: str = "ip",
):
    """
    Sample a subset of data and find out the nearest neighbors index

    Args:
      data: numpy array of shape (N, d)
      k: number of nearest neighbors
      query: numpy array of shape (M, d)
      sample_ratio: ratio of data to sample
      seed: random seed
      device: 'cpu' or 'cuda'
      normalize: normalize data or not
      distance: 'ip' for inner product, 'l2' for l2 distance
    """

    # check distance
    if distance not in ["ip", "l2"]:
        raise ValueError(f"Unsupported distance: {distance}")

    if normalize:
        data = data / np.linalg.norm(data, axis=1, keepdims=True)
        if query is not None:
            query = query / np.linalg.norm(query, axis=1, keepdims=True)

    if query is None:
        np.random.seed(seed)
        N = data.shape[0]
        sample_size = int(N * sample_ratio)
        sample_idx = np.random.choice(N, sample_size, replace=False)
        print(f"sample_idx: {sample_idx}")
        query = data[sample_idx]

    query = torch.from_numpy(query).float()
    data = torch.from_numpy(data).float()
    if device != "cpu":
        query = query.to(device)
        data = data.to(device)

    nn_idx_all = torch.empty((query.shape[0], k), dtype=torch.long)
    step_size = 1000
    for i in tqdm.trange(0, sample_size, step_size):
        if distance == "ip":
            dist = torch.mm(query[i : i + step_size], data.t())
        else:
            # distance == 'l2':
            dist = torch.cdist(query[i : i + step_size], data)

        if distance == "ip":
            _, nn_idx = torch.topk(dist, k=k, dim=1, largest=True)
        else:
            # distance == 'l2':
            _, nn_idx = torch.topk(-dist, k=k, dim=1, largest=True)

        nn_idx_all[i : i + step_size] = nn_idx

    print(f"dist_all shape: {nn_idx_all.shape}")
    print(nn_idx_all)

    return query.numpy(), nn_idx_all.numpy()
