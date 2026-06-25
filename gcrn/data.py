from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class StandardScaler:
    mean: float
    std: float

    def transform(self, values):
        return (values - self.mean) / self.std

    def inverse_transform(self, values):
        return values * self.std + self.mean


def load_adjacency(csv_path: Path, num_nodes: int) -> torch.Tensor:
    edges = pd.read_csv(csv_path)
    adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    sigma = max(float(edges["cost"].std()), 1e-6)
    for src, dst, cost in edges[["from", "to", "cost"]].itertuples(index=False):
        weight = np.exp(-((float(cost) / sigma) ** 2))
        adj[int(src), int(dst)] = weight
        adj[int(dst), int(src)] = weight

    adj += np.eye(num_nodes, dtype=np.float32)
    degree = np.maximum(adj.sum(axis=1), 1e-6)
    degree_inv_sqrt = degree ** -0.5
    adj = degree_inv_sqrt[:, None] * adj * degree_inv_sqrt[None, :]
    return torch.from_numpy(adj)


def make_windows(values, input_steps, output_steps):
    xs, ys = [], []
    window_count = len(values) - input_steps - output_steps + 1
    for start in range(window_count):
        split = start + input_steps
        xs.append(values[start:split])
        ys.append(values[split:split + output_steps])
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)


def load_dataset(config):
    root = Path(config["data_dir"])
    name = config["dataset"]
    raw = np.load(root / f"{name}.npz")["data"].astype(np.float32)
    # PEMS stores flow, occupancy and speed in the last dimension.
    flow = raw[..., :1]
    length = len(flow)
    train_end = int(length * config["train_ratio"])
    val_end = int(length * (config["train_ratio"] + config["val_ratio"]))
    scaler = StandardScaler(
        mean=float(flow[:train_end].mean()),
        std=max(float(flow[:train_end].std()), 1e-6),
    )
    flow = scaler.transform(flow)

    overlap = config["input_steps"] + config["output_steps"] - 1
    split_values = {
        "train": flow[:train_end],
        "val": flow[max(0, train_end - overlap):val_end],
        "test": flow[max(0, val_end - overlap):],
    }
    loaders = {}
    for split, values in split_values.items():
        x, y = make_windows(values, config["input_steps"], config["output_steps"])
        loaders[split] = DataLoader(
            TensorDataset(torch.from_numpy(x), torch.from_numpy(y)),
            batch_size=config["batch_size"],
            shuffle=split == "train",
            pin_memory=torch.cuda.is_available(),
        )
    adj = load_adjacency(root / f"{name}.csv", flow.shape[1])
    return loaders, scaler, adj
