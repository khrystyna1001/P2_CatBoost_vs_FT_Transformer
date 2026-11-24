"""Training loop for Part B (DL S1-S3): AdamW, early stopping on validation ROC-AUC.

The loop is shared by the MLP and the FT-Transformer and is used by the M3
baseline, the Optuna tuning (M4) and the final multi-seed runs.
"""

from __future__ import annotations

import copy
import random
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from data.datasets import Dataset
from data.splits import load_split
from src.eval.metrics import roc_auc
from src.part_b.models import build_model
from src.part_b.preprocessing import TabularPreprocessor

torch.set_num_threads(2)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@dataclass
class Tensors:
    x_num: torch.Tensor
    x_cat: torch.Tensor
    y: torch.Tensor


@dataclass
class PreparedData:
    pre: TabularPreprocessor
    train: Tensors
    val: Tensors
    test: Tensors | None
    splits: dict


def prepare(ds: Dataset, min_count: int = 10, include_test: bool = False) -> PreparedData:
    """Fit pre-processing on training rows and transform all splits."""
    sp = load_split(ds.name)
    X, y = ds.X, ds.y
    pre = TabularPreprocessor(ds.num_cols, ds.cat_cols, min_count=min_count).fit(X.iloc[sp["train"]])

    def tens(idx):
        xn, xc = pre.transform(X.iloc[idx])
        return Tensors(torch.from_numpy(xn), torch.from_numpy(xc), torch.from_numpy(y[idx].astype(np.float32)))

    return PreparedData(pre, tens(sp["train"]), tens(sp["val"]), tens(sp["test"]) if include_test else None, sp)


@torch.no_grad()
def predict(model: torch.nn.Module, t: Tensors, batch_size: int = 4096) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(t.y), batch_size):
        out.append(torch.sigmoid(model(t.x_num[i:i + batch_size], t.x_cat[i:i + batch_size])))
    return torch.cat(out).numpy()


@dataclass
class TrainResult:
    model: torch.nn.Module
    best_val_auc: float
    best_epoch: int
    history: pd.DataFrame
    train_time_s: float
    extra: dict = field(default_factory=dict)


def train(kind: str, params: dict, data: PreparedData, seed: int, max_epochs: int = 100, patience: int = 10,
          time_budget_s: float | None = None, verbose: bool = False, model=None, on_epoch=None) -> TrainResult:
    set_seed(seed)
    if model is None:
        model = build_model(kind, data.train.x_num.shape[1], data.pre.cardinalities, params)
    opt = torch.optim.AdamW(model.parameters(), lr=params.get("lr", 1e-4), weight_decay=params.get("weight_decay", 1e-5))
    bs = int(params.get("batch_size", 256))
    tr = data.train
    n = len(tr.y)
    g = torch.Generator().manual_seed(seed)

    best_auc, best_epoch, best_state, wait = -np.inf, -1, None, 0
    hist = []
    t0 = time.perf_counter()
    for epoch in range(max_epochs):
        model.train()
        perm = torch.randperm(n, generator=g)
        tot = 0.0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            logit = model(tr.x_num[idx], tr.x_cat[idx])
            loss = F.binary_cross_entropy_with_logits(logit, tr.y[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)

        p_val = predict(model, data.val)
        yv = data.val.y.numpy()
        val_loss = float(F.binary_cross_entropy(torch.from_numpy(p_val).clamp(1e-7, 1 - 1e-7), data.val.y).item())
        val_auc = roc_auc(yv, p_val)
        hist.append(dict(epoch=epoch, train_loss=tot / n, val_loss=val_loss, val_auc=val_auc,
                         elapsed_s=time.perf_counter() - t0))
        if verbose:
            print(f"  epoch {epoch:3d} train_loss={tot / n:.4f} val_loss={val_loss:.4f} val_auc={val_auc:.4f}", flush=True)
        if on_epoch is not None:
            on_epoch(epoch, val_auc)  # may raise optuna.TrialPruned
        if val_auc > best_auc:
            best_auc, best_epoch, wait = val_auc, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            wait += 1
        if wait >= patience:
            break
        if time_budget_s is not None and time.perf_counter() - t0 > time_budget_s:
            break

    model.load_state_dict(best_state)
    model.eval()
    return TrainResult(model, float(best_auc), best_epoch, pd.DataFrame(hist), time.perf_counter() - t0)
