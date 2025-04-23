import argparse
import os
import numpy as np
from hakes.index.build import (
    build_dataset,
    init_hakes_params,
    train_hakes_params,
    recenter_ivf,
)
from hakes.index.dataset import load_data_bin


def parse_args():
    parser = argparse.ArgumentParser(description="Build Hakes index")
    parser.add_argument("--N", type=int, help="Number of data points", required=True)
    parser.add_argument("--d", type=int, help="Dimension of data points", required=True)
    parser.add_argument(
        "--dr", type=int, help="Dimension reduction output dimension", required=True
    )
    parser.add_argument(
        "--nlist", type=int, help="Number of clusters in IVF", required=True
    )
    parser.add_argument(
        "--metric",
        type=str,
        help="Metric to use for the index",
        choices=["ip", "l2"],
        default="ip",
    )
    parser.add_argument(
        "--epoch", type=int, help="Number of epochs to train the index", default=1
    )
    parser.add_argument(
        "--batch_size", type=int, help="Batch size for training the index", default=512
    )
    parser.add_argument(
        "--vt_lr", type=float, help="Learning rate for vector transform", default=1e-5
    )
    parser.add_argument(
        "--pq_lr",
        type=float,
        help="Learning rate for product quantization",
        default=1e-5,
    )
    parser.add_argument(
        "--lamb",
        type=float,
        help="Control the weight of vt loss, -1 to rescale against pq loss",
        default=-1,
    )
    parser.add_argument(
        "--device",
        type=str,
        help="Device to run the training, for example, cuda:0",
        default="cpu",
    )
    parser.add_argument(
        "--data_path", type=str, help="Path to data binary file", required=True
    )
    parser.add_argument(
        "--query_path",
        type=str,
        help="Path to query binary file, if not provided, use data_path",
        default=None,
    )
    parser.add_argument("--query_n", type=int, help="Number of query points", default=0)
    parser.add_argument(
        "--output_path", type=str, help="Path to save the index", required=True
    )
    parser.add_argument(
        "--use_query_recenter",
        action="store_true",
        help="Use query recentering",
    )

    return parser.parse_args()


def run(args):
    data = load_data_bin(args.data_path, args.N, args.d)
    index = init_hakes_params(data, args.dr, args.nlist, args.metric)
    index.set_fixed_assignment(True)

    os.makedirs(args.output_path, exist_ok=True)
    index.save_as_hakes_index(os.path.join(args.output_path, "findex.bin"))
    sample_ratio = 1
    if args.N > 256 * 256:
        sample_ratio = 256 * 256 / args.N
    if args.query_path is not None:
        query = load_data_bin(args.query_path, args.query_n, args.d)
        dataset = build_dataset(data, query, sample_ratio=sample_ratio, nn=50)
    else:
        dataset = build_dataset(data, sample_ratio=sample_ratio, nn=50)
    train_hakes_params(
        model=index,
        dataset=dataset,
        epochs=args.epoch,
        batch_size=args.batch_size,
        lr_params={"vt": args.vt_lr, "pq": args.pq_lr, "ivf": 0},
        loss_weight={
            "vt": args.lamb if args.lamb != -1 else "rescale",
            "pq": 1,
            "ivf": 0,
        },
        temperature=1,
        loss_method="hakes",
        device=args.device,
    )
    print("Recenter IVF with data")
    recenter_ivf(index, dataset.query, 1, args.metric)
    index.save_as_hakes_index(os.path.join(args.output_path, "uindex.bin"))


if __name__ == "__main__":
    args = parse_args()
    print(args)
    run(args)
