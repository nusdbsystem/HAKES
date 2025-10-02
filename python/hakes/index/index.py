import logging
import os
import torch
from torch import nn
import torch.nn.functional as F
from typing import Type

from .vt import HakesPreTransform
from .ivf import HakesIVF
from .pq import HakesPQ


class HakesIndex(nn.Module):
    def __init__(
        self,
        vts: Type[HakesPreTransform] = None,
        ivf: Type[HakesIVF] = None,
        pq: Type[HakesPQ] = None,
        metric: str = "ip",
    ):
        super().__init__()
        self.metric = metric
        self.vts = vts
        self.fixed_pretransform = None
        self.ivf = ivf
        self.pq = pq

        self.fixed_assignment = False

    def set_fixed_assignment(self, fixed_assignment: bool):
        self.fixed_assignment = fixed_assignment
        if fixed_assignment:
            if self.pq is not None:
                self.pq.set_fixed_assignment(fixed_assignment)
            self.fixed_pretransform = self.vts.clone()
        else:
            self.fixed_pretransform = None

    @classmethod
    def load(cls, path: str, metric: str):
        assert path, "init_index_file must be provided"

        logging.info(f"Loading index of from {path}")

        if not os.path.exists(path):
            raise ValueError(f"Index file {path} does not exist")
        else:
            logging.info("Loading from hakes index file")
            return HakesIndex.load_from_hakes_index(path, metric=metric)

    @classmethod
    def load_from_hakes_index(cls, path: str, metric="ip"):
        with open(path, "rb") as f:
            vts = HakesPreTransform.from_reader(f)
            ivf = HakesIVF.from_reader(f)
            pq = HakesPQ.from_reader(f)
            assert ivf.metric == metric
            return cls(vts, ivf, pq, metric=metric)

    def __str__(self):
        vts_str = f"VecTransform: {self.vts}\n" if self.vts is not None else ""
        ivf_str = f"IVF: {self.ivf}\n" if self.ivf is not None else ""
        pq_str = f"PQ: {self.pq}\n" if self.pq is not None else ""
        return super().__str__() + vts_str + ivf_str + pq_str + f"metric: {self.metric}"

    def compute_score(self, q, cands, metric):
        # removed the temperature argument
        assert metric in ["ip", "l2"]
        assert q.shape[-1] == cands.shape[-1]

        if metric == "ip":
            if len(q.shape) == 2 and len(cands.shape) == 2:
                return torch.matmul(q, cands.T)
            elif len(q.shape) == 3 and len(cands.shape) == 3:
                return torch.matmul(q, torch.swapaxes(cands, -2, -1)).squeeze(1)
            elif len(q.shape) == 2 and len(cands.shape) == 3:
                return torch.matmul(
                    q[:, None, :], torch.swapaxes(cands, -2, -1)
                ).squeeze(1)
            else:
                raise ValueError(f"Unknown shape: {q.shape}, {cands.shape}")
        else:  # l2
            if len(q.shape) == 2 and len(cands.shape) == 2:
                return torch.cdist(q, cands, p=2)
            elif len(q.shape) == 3 and len(cands.shape) == 3:
                return torch.cdist(q, cands, p=2).squeeze(1)
            elif len(q.shape) == 2 and len(cands.shape) == 3:
                return torch.cdist(q[:, None, :], cands, p=2).squeeze(1)
            else:
                raise ValueError(f"Unknown shape: {q.shape}, {cands.shape}")

    def kldiv_loss(self, target: torch.Tensor, score: torch.Tensor):
        if target is None or score is None:
            logging.warning("Target or score is None")
            return 0
        target_smax = F.softmax(target, dim=-1)
        return F.kl_div(
            F.log_softmax(score, dim=-1), target_smax, reduction="batchmean"
        )

    def mse_loss(self, teacher_score: torch.Tensor, student_score: torch.Tensor):
        if teacher_score is None or student_score is None:
            logging.warning("Teacher or student score is None")
            return 0
        return F.mse_loss(teacher_score, student_score)

    def recons_loss(self, data: torch.Tensor, quantized: torch.Tensor):
        if data is None or quantized is None:
            return 0
        return F.mse_loss(data, quantized)

    def forward(
        self,
        query_data: torch.FloatTensor,
        pos_data: torch.FloatTensor,
        temperature: float = 1,
        loss_method: str = "hakes",
    ):
        pos_data = torch.squeeze(pos_data, 1)

        # rotation if needed

        if self.vts is not None:
            transformed_query_data = self.vts(query_data)
            transformed_pos_data = self.vts(pos_data)
        else:
            transformed_query_data = query_data
            transformed_pos_data = pos_data

        logging.info(
            f"transformed shape: {transformed_query_data.shape}, {transformed_pos_data.shape}"
        )

        if self.pq is not None:
            if self.ivf is not None and self.ivf.by_residual:
                raise ValueError("PQ is not supported with IVF by_residual")
            elif self.fixed_assignment:
                query_quantized = self.pq.quantization(transformed_query_data)
                pos_quantized = self.pq.quantization(self.fixed_pretransform(pos_data))
            else:
                query_quantized = self.pq.quantization(transformed_query_data)
                pos_quantized = self.pq.quantization(transformed_pos_data)
        else:
            query_quantized = None
            pos_quantized = None

        origin_pos_score = self.compute_score(query_data, pos_data, self.metric)
        vt_pos_score = self.compute_score(
            transformed_query_data,
            transformed_pos_data,
            self.metric,
        )
        pq_pos_score = self.compute_score(
            transformed_query_data, pos_quantized, self.metric
        )

        vt_loss, ivf_loss, pq_loss = 0, 0, 0

        if loss_method == "hakes":
            vt_loss = self.kldiv_loss(origin_pos_score, vt_pos_score)
            if self.fixed_pretransform:
                # print(f"pq use kldiv loss: shape: {origin_pos_score.shape}, {pq_pos_score.shape}")
                pq_loss = self.kldiv_loss(origin_pos_score, pq_pos_score)
            else:
                pq_loss = self.recons_loss(
                    transformed_pos_data, self.pq.quantization(transformed_pos_data)
                )
        else:
            raise ValueError(f"Unknown loss method: {loss_method}")
        logging.info(
            f"pq query recons loss: {self.recons_loss(transformed_query_data, query_quantized)}"
        )

        logging.info(f"losses: {vt_loss}, {pq_loss}, {ivf_loss}, ")

        return vt_loss, pq_loss, ivf_loss

    def report_distance(
        self, query_data: torch.FloatTensor, neighbor_data: torch.FloatTensor, mode: str
    ):
        if mode == "raw":
            return self.compute_score(query_data, neighbor_data, self.metric)
        elif mode == "vt":
            transformed_query_data = self.vts(query_data)
            transformed_neighbor_data = self.vts(neighbor_data)
            logging.info(
                f"transformed shape: {transformed_query_data.shape}, {transformed_neighbor_data.shape}"
            )
            # calculate distances
            return self.compute_score(
                transformed_query_data, transformed_neighbor_data, self.metric
            )
        elif mode == "ivf":
            neighbor_centers = self.ivf.select_centers(self.vts(neighbor_data))
            return self.compute_score(
                self.vts(query_data), neighbor_centers, self.metric
            )
        elif mode == "pq":
            if self.fixed_assignment:
                neighbor_quantized = self.pq.quantization(
                    self.fixed_pretransform(neighbor_data)
                )
            else:
                neighbor_quantized = self.pq.quantization(self.vts(neighbor_data))
            return self.compute_score(
                self.vts(query_data), neighbor_quantized, self.metric
            )
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def save_as_hakes_index(self, path: str):
        with open(path, "wb") as f:
            if self.vts is not None:
                self.vts.save_to_writer(f)
            if self.ivf is not None:
                self.ivf.save_to_writer(f)
            if self.pq is not None:
                self.pq.save_to_writer(f)
