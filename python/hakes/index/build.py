import numpy as np
import torch
from torch.utils.data import RandomSampler, DataLoader
from torch.optim.adamw import AdamW
from tqdm import trange
import time

from typing import Dict

from .dataset import HakesDataset, HakesDataCollator
from .index import HakesIndex
from .ivf import kmeans_ivf
from .nn import get_nn
from .opq import opq_train
from .vt import HakesPreTransform, HakesVecTransform
from .pq import HakesPQ

"""
  HAKES training loops through the sampled vector - ANN pairs to update index parameters.
"""


def build_dataset(
    data: np.ndarray,
    query: np.ndarray = None,
    pos_ids: np.ndarray = None,
    sample_ratio: float = 0.01,
    nn: int = 50,
):
    if query is None or pos_ids is None:
        query, pos_ids = get_nn(data, nn, query, sample_ratio, 0)
    return HakesDataset(data, query, pos_ids)


def init_hakes_params(
    data: np.ndarray,
    vt_out_d: int,
    nlist: int,
    metric: str = "ip",
):
    """
    Initialize Hakes index parameters

    Args:
      data: numpy array of shape (N, d)
    """
    if vt_out_d % 2 != 0:
        raise ValueError("vt_out_d must be divisible by 2")
    # if data is over 256*256 samples then we need to sample
    # 256*256 samples to train the OPQ
    if data.shape[0] > 256 * 256:
        data = data[np.random.choice(data.shape[0], 256 * 256, replace=False)]
    print(f"sampled data shape for opq: {data.shape}")

    # OPQ
    A, pq = opq_train(data, vt_out_d, vt_out_d // 2, iter=20)
    projected_data = data @ A
    # IVF
    ivf_start = time.time()
    ivf = kmeans_ivf(projected_data, nlist, niter=20)
    print(f"IVF training time: {time.time() - ivf_start}")

    # HakesIndex
    d = data.shape[1]
    return HakesIndex(
        HakesPreTransform(
            torch.nn.ModuleList(
                [HakesVecTransform(d, vt_out_d, A.T, np.zeros(vt_out_d))]
            )
        ),
        ivf,
        HakesPQ(vt_out_d, vt_out_d // 2, nbits=4, codebook=pq, fixed_assignment=True),
        metric,
    )


def train_hakes_params(
    model: HakesIndex,
    dataset: HakesDataset,
    epochs: int = 10,
    batch_size: int = 512,
    warmup_steps_ratio: float = 0.1,
    lr_params: Dict[str, float] = {"vt": 1e-4, "pq": 1e-4, "ivf": 1e-4},
    loss_weight: Dict[str, float] = {"vt": 0.0, "pq": 1.0, "ivf": 1.0},
    temperature: float = 1.0,
    loss_method: str = "hakes",
    max_grad_norm: float = -1,
    device="cpu",
    checkpoint_path="./checkpoint",
):
    sampler = RandomSampler(dataset)
    dataloader = DataLoader(
        dataset, sampler=sampler, batch_size=batch_size, collate_fn=HakesDataCollator()
    )
    model.to(device)
    num_train_steps = len(dataloader) * epochs

    # setup optimizer
    param_optimizer = list(model.named_parameters())
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in param_optimizer if "vts" in n and "ivf" not in n],
            "weight_decay": 0.999,
            "lr": lr_params["vt"],
        },
        {
            "params": [p for n, p in param_optimizer if "ivf" in n and "vts" not in n],
            "weight_decay": 0.999,
            "lr": lr_params["ivf"],
        },
        {
            "params": [p for n, p in param_optimizer if "pq.codebooks" in n],
            "weight_decay": 0.999,
            "lr": lr_params["pq"],
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters)

    # setup learning rate scheduler
    num_warmup_steps = int(num_train_steps * warmup_steps_ratio)
    lr_lambda = lambda step: (
        float(step) / float(max(1, num_warmup_steps))
        if step < num_warmup_steps
        else max(
            0.0,
            float(num_train_steps - step)
            / float(max(1, num_train_steps - num_warmup_steps)),
        )
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    # train
    for epoch in trange(epochs, desc="Epoch"):
        model.train()
        for step, sample in enumerate(dataloader):
            # to update, we are not using dict
            query_data = sample["query_data"].to(device)
            pos_data = sample["pos_data"].to(device)
            # check the effect of fix_emb='doc', cross_device_sample=True
            batch_vt_loss, batch_pq_loss, batch_ivf_loss = model(
                query_data=query_data,
                pos_data=pos_data,
                temperature=temperature,
                loss_method=loss_method,
            )
            vt_rescale = (
                (
                    (batch_pq_loss / batch_vt_loss).item() * loss_weight["pq"]
                    if batch_vt_loss != 0.0
                    else 0
                )
                if loss_weight["vt"] == "rescale"
                else float(loss_weight["vt"])
            )
            batch_loss = (
                vt_rescale * batch_vt_loss
                + loss_weight["pq"] * batch_pq_loss
                + loss_weight["ivf"] * batch_ivf_loss
            )
            batch_loss.backward()
            print(
                f"batch loss = {vt_rescale} * {batch_vt_loss} + {loss_weight['pq']} * {batch_pq_loss} + {loss_weight['ivf']} * {batch_ivf_loss}"
            )
            print(f"epoch {epoch} step {step} batch_loss: {batch_loss}")

            if max_grad_norm != -1:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()

            optimizer.zero_grad()
            model.ivf.normalize_centers()


def recenter_ivf(
    model: HakesIndex,
    data: np.ndarray,
    sample_ratio: float = 0.01,
):
    # perform recenter just on cpu
    model.to("cpu")
    recenter_indices = np.random.choice(
        data.shape[0], int(data.shape[0] * sample_ratio), replace=False
    )
    recenter_data = data[recenter_indices]
    grouped_vectors = [[] for _ in range(model.ivf.nlist)]
    batch_size = 1024 * 10
    for i in trange(0, recenter_data.shape[0], batch_size):
        batch = model.vts(
            torch.tensor(recenter_data[i : min(i + batch_size, recenter_data.shape[0])])
        )
        assignment = model.ivf.get_assignment(batch)
        for j in range(batch.shape[0]):
            grouped_vectors[assignment[j].item()].append(i + j)
    new_centers = []
    for i in range(model.ivf.nlist):
        vecs = recenter_data[grouped_vectors[i]]
        vecs_tensor = torch.tensor(vecs)
        vt_vecs = model.vts(vecs_tensor)
        rep = torch.mean(vt_vecs, dim=0)
        normalized_rep = rep / torch.norm(rep)
        new_centers.append(normalized_rep.detach().cpu().numpy())
    model.ivf.update_centers(np.array(new_centers))
    return model
