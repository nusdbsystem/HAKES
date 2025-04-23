import copy
import io
import logging
import numpy as np
import struct
import torch

from .debug import matrix_to_str


class HakesVecTransform(torch.nn.Module):
    def __init__(
        self,
        d_in: int,
        d_out: int,
        A: np.ndarray,
        b: np.ndarray,
    ):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.A = torch.nn.Parameter(
            torch.FloatTensor(copy.deepcopy(A)), requires_grad=True
        )
        if b.shape[0] == 0:
            self.b = torch.nn.Parameter(torch.zeros(d_out), requires_grad=True)
        else:
            self.b = torch.nn.Parameter(
                torch.FloatTensor(copy.deepcopy(b)), requires_grad=True
            )
        logging.info(f"Initialized VecTransform with d_in: {d_in}, d_out: {d_out}")

    def clone(self):
        return HakesVecTransform(
            self.d_in,
            self.d_out,
            self.A.detach().cpu().numpy(),
            self.b.detach().cpu().numpy(),
        )

    def forward(self, x):
        return torch.torch.nn.functional.linear(x, self.A, self.b)


class HakesPreTransform(torch.nn.Module):
    def __init__(
        self,
        vt_list: torch.nn.ModuleList,
    ):
        super().__init__()
        self.vt_list = vt_list
        logging.info(f"Initialized PreTransform with {len(vt_list)} VecTransforms")

    def __str__(self):
        vts_str = ""
        for i, vt in enumerate(self.vt_list):
            vts_str += f"VecTransform {i}: {vt}\nA: {matrix_to_str(vt.A)} b: {matrix_to_str(vt.b)}\n"
        return (
            super().__str__() + f" with {len(self.vt_list)} VecTransforms\n" + vts_str
        )

    def clone(self):
        vt_list = torch.nn.ModuleList()
        for vt in self.vt_list:
            vt_list.append(vt.clone())
        return HakesPreTransform(vt_list)

    @classmethod
    def from_reader(cls, reader: io.BufferedReader):
        n = struct.unpack("<i", reader.read(4))[0]
        vt_list = torch.nn.ModuleList()
        for _ in range(n):
            d_out = struct.unpack("<i", reader.read(4))[0]
            d_in = struct.unpack("<i", reader.read(4))[0]
            A = np.frombuffer(reader.read(d_out * d_in * 4), dtype="<f").reshape(
                d_out, d_in
            )
            b = np.frombuffer(reader.read(d_out * 4), dtype="<f")
            vt_list.append(HakesVecTransform(d_in, d_out, A, b))
        return cls(vt_list)

    def reduce_dim(self, target_d) -> bool:
        """
        we reduce the first transformation dimension
        return if the downstream operator need to reduce dim
        """

        if len(self.vt_list) == 0:
            logging.info("No VecTransform in the PreTransform, skip reducing dim")
            return False

        if target_d > self.vt_list[0].d_in or target_d > self.vt_list[0].d_out:
            raise ValueError(
                f"target_d {target_d} must be less than first vt d_in {self.vt_list[0].d_in} and d_out {self.vt_list[0].d_out}"
            )
        self.vt_list[0].d_out = target_d
        self.vt_list[0].A.requires_grad = False
        self.vt_list[0].b.requires_grad = False
        self.vt_list[0].A = torch.nn.Parameter(
            self.vt_list[0].A[:target_d, :], requires_grad=True
        )
        self.vt_list[0].b = torch.nn.Parameter(
            self.vt_list[0].b[:target_d], requires_grad=True
        )
        for vt in self.vt_list[1:]:
            vt.d_in = target_d
            vt.d_out = target_d
            vt.A.requires_grad = False
            vt.b.requires_grad = False
            vt.A = torch.nn.Parameter(vt.A[:target_d, :target_d], requires_grad=True)
            vt.b = torch.nn.Parameter(vt.b[:target_d], requires_grad=True)
        return True

    def forward(self, x):
        for vt in self.vt_list:
            x = vt(x)
        return x

    def save_to_writer(self, writer: io.BufferedWriter):
        """
        format: (little endian)
        n: int32 - number of vector transform
        vector transforms:
            d_out: int32
            d_in: int32
            A: float32 array
            b: float32 array
        """
        logging.info("Saving PreTransform")
        writer.write(struct.pack("<i", len(self.vt_list)))
        for vt in self.vt_list:
            writer.write(struct.pack("<i", vt.d_out))
            writer.write(struct.pack("<i", vt.d_in))
            writer.write(
                np.ascontiguousarray(vt.A.detach().cpu().numpy(), dtype="<f").tobytes()
            )
            writer.write(
                np.ascontiguousarray(vt.b.detach().cpu().numpy(), dtype="<f").tobytes()
            )
