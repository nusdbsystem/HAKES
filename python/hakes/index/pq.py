import copy
import io
import logging
import numpy as np
import struct
import torch

from .debug import codebook_to_str


class HakesPQ(torch.nn.Module):
    def __init__(
        self,
        d: int,
        m: int,
        nbits: int = 4,
        codebook: np.ndarray = None,
        fixed_assignment: bool = False,
    ):
        super().__init__()
        if codebook is not None:
            assert codebook.shape == (m, 2**nbits, d // m)
            self.assignment_codebooks = torch.nn.Parameter(
                torch.FloatTensor(copy.deepcopy(codebook)), requires_grad=False
            )
            self.codebooks = torch.nn.Parameter(
                torch.FloatTensor(copy.deepcopy(codebook)), requires_grad=True
            )
        else:
            self.codebooks = torch.nn.Parameter(
                torch.empty(m, 2**nbits, d // m)
                .uniform_(-0.1, 0.1)
                .type(torch.FloatTensor),
                requires_grad=True,
            )

        self.d = d
        self.m = m
        self.dsub = d // m
        self.ksub = 1 << nbits
        self.nbits = nbits

        self.fixed_assignment = fixed_assignment

        logging.info(f"Initialized PQ with d: {d}, m: {m}, nbits: {nbits}")

    def set_fixed_assignment(self, fixed_assignment: bool):
        logging.info(f"Set fixed assignment to {fixed_assignment}")
        self.fixed_assignment = fixed_assignment

    def __str__(self):
        return (
            super().__str__()
            + f" d: {self.d}, m: {self.m}, nbits: {self.nbits}\n"
            + codebook_to_str(self.codebooks)
        )

    @classmethod
    def from_reader(cls, reader: io.BufferedReader):
        d = struct.unpack("<i", reader.read(4))[0]
        m = struct.unpack("<i", reader.read(4))[0]
        nbits = struct.unpack("<i", reader.read(4))[0]
        codebooks = np.frombuffer(
            reader.read(m * 2**nbits * d // m * 4), dtype="<f"
        ).reshape(m, 2**nbits, d // m)
        return cls(d, m, nbits, codebooks)

    def reduce_dim(self, target_d):
        if target_d % self.dsub != 0:
            raise ValueError(
                f"target_d ({target_d}) must be a multiple of dsub ({self.dsub})"
            )

        self.d = target_d
        self.m = target_d // self.dsub
        self.codebooks.requires_grad = False

        self.codebooks = torch.nn.Parameter(
            self.codebooks[: self.m, :, : target_d // self.m], requires_grad=True
        )

    def quantization(self, vecs):
        # generate quantized codes assignment probability
        batch_shape = vecs.shape
        vecs = vecs.view(-1, self.m, self.dsub)  # (n, m, d/m)
        codebook = self.codebooks.unsqueeze(0).expand(
            vecs.size(0), -1, -1, -1
        )  # (n, m, 2**nbits, d/m)

        if self.fixed_assignment:
            assign_codebook = self.assignment_codebooks.unsqueeze(0).expand(
                vecs.size(0), -1, -1, -1
            )  # (n, m, 2**nbits, d/m)
            # PQ code assignment always uses l2 distance for assignment.
            assign_prob = -torch.sum(
                (vecs.unsqueeze(-2) - assign_codebook) ** 2, -1
            )  # (n, m, 2**nbits)
        else:
            assign_prob = -torch.sum((vecs.unsqueeze(-2) - codebook) ** 2, -1)
        assign = torch.nn.functional.softmax(assign_prob, dim=-1)  # (n, m, 2**nbits)

        # generate hard assignment (STE)
        # use STE rather than Gumbel-Softmax because the distance value are close and even small temperature like 0.01 would not make a clear assignment.
        index = assign.max(dim=-1, keepdim=True)[1]  # (n, m, 1) [1] here is the index
        assign_hard = torch.zeros_like(
            assign, device=assign.device, dtype=assign.dtype
        ).scatter_(
            -1, index, 1.0
        )  # (n, m, 2**nbits)
        assign = assign_hard.detach() - assign.detach() + assign  # (n, m, 2**nbits)

        # generate quantized codes
        assign = assign.unsqueeze(2)  # (n, m, 1, 2**nbits)
        quantized_vecs = torch.matmul(assign, codebook).squeeze(2)  # (n, m, d/m)
        quantized_vecs = quantized_vecs.view(batch_shape)  # (n, d)
        return quantized_vecs

    def save_to_writer(self, writer: io.BufferedWriter):
        """
        format: (little endian)
        d: int32
        m: int32
        nbits: int32
        codebooks: float32 array of shape (m, 2**nbits, d // m)
        """
        logging.info(f"Saving PQ")
        writer.write(struct.pack("<i", self.d))
        writer.write(struct.pack("<i", self.m))
        writer.write(struct.pack("<i", self.nbits))
        writer.write(
            np.ascontiguousarray(
                self.codebooks.detach().cpu().numpy(), dtype="<f"
            ).tobytes()
        )
