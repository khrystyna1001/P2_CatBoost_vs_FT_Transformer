"""M5 — export the tuned FT-Transformer (seed 0) of each dataset to ONNX.

The ONNX graph has two outputs: ``logit`` (batch,) and ``cls_attention``
(batch, n_features), the [CLS]-to-feature attention averaged over heads and
layers, used by the API's /explain endpoint for Part B.

Also writes models/<dataset>/meta.json with the training-data hash, model file
hashes and test metrics, read by the API's /model-info endpoint.

Checks:
  * ONNX Runtime and PyTorch outputs agree (max |diff| < 1e-4 on 1 000 test rows)
  * an unseen category is mapped to index 0 by the serving pre-processor

Run from repo root:  uv run python src/part_b/export_onnx.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from data.datasets import load_dataset
from data.splits import load_split
from src.part_b.models import build_model
from src.part_b.preprocessing import UNK, TabularPreprocessor

DATASETS = ["adult", "bank-marketing", "amazon_employee_access"]


class ExportWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x_num, x_cat):
        logit, att = self.model(x_num, x_cat, return_attention=True)
        return logit, att[:, :, :, 0, 1:].mean(dim=(1, 2))


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def training_data_hash(ds, train_idx) -> str:
    X = ds.X.iloc[train_idx]
    h = hashlib.sha256(pd.util.hash_pandas_object(X, index=False).values.tobytes())
    h.update(ds.y[train_idx].tobytes())
    return h.hexdigest()


def export(ds_name: str) -> dict:
    mdir = ROOT / "models" / ds_name
    spec = json.loads((mdir / "ft_transformer.json").read_text())
    model = build_model("ft_transformer", spec["n_num"], spec["cardinalities"], spec["params"])
    model.load_state_dict(torch.load(mdir / "ft_transformer.pt"))
    model.eval()
    pre = TabularPreprocessor.load(mdir / "preprocessor.json")

    ds = load_dataset(ds_name)
    sp = load_split(ds_name)
    X_te = ds.X.iloc[sp["test"][:1000]]
    xn, xc = pre.transform(X_te)
    xn_t, xc_t = torch.from_numpy(xn), torch.from_numpy(xc)

    wrapper = ExportWrapper(model).eval()
    onnx_path = mdir / "ft_transformer.onnx"
    torch.onnx.export(
        wrapper, (xn_t[:2], xc_t[:2]), str(onnx_path), input_names=["x_num", "x_cat"],
        output_names=["logit", "cls_attention"],
        dynamic_axes={"x_num": {0: "batch"}, "x_cat": {0: "batch"}, "logit": {0: "batch"}, "cls_attention": {0: "batch"}},
        opset_version=17, dynamo=False,
    )

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    # a data set without numerical features (Amazon) has no x_num input in the graph
    names = {i.name for i in sess.get_inputs()}
    o_logit, o_att = sess.run(None, {k: v for k, v in (("x_num", xn), ("x_cat", xc)) if k in names})
    with torch.no_grad():
        t_logit, t_att = wrapper(xn_t, xc_t)
    diff = float(max(np.abs(o_logit - t_logit.numpy()).max(), np.abs(o_att - t_att.numpy()).max()))
    assert diff < 1e-4, f"ONNX mismatch {diff}"

    # unseen categories must map to index 0
    probe = X_te.iloc[:1].copy()
    for c in pre.cat_cols:
        probe[c] = "__never_seen_category__"
    assert (pre.transform_cat(probe) == UNK).all()

    scores = pd.read_csv(ROOT / "results" / "m4_test_scores.csv")
    s = scores[scores.dataset == ds_name].set_index("model")
    meta = dict(
        dataset=ds_name,
        training_data_sha256=training_data_hash(ds, sp["train"]),
        n_train_rows=int(len(sp["train"])),
        features=dict(numerical=pre.num_cols, categorical=pre.cat_cols),
        models={
            "part_a": dict(name="catboost", file="catboost.cbm", sha256=sha256(mdir / "catboost.cbm"),
                           test_roc_auc=float(s.loc["catboost", "test_auc_mean"]),
                           test_avg_precision=float(s.loc["catboost", "test_ap_mean"])),
            "part_b": dict(name="ft_transformer", file="ft_transformer.onnx", sha256=sha256(onnx_path),
                           test_roc_auc=float(s.loc["ft_transformer", "test_auc_mean"]),
                           test_avg_precision=float(s.loc["ft_transformer", "test_ap_mean"]),
                           onnx_max_abs_diff_vs_torch=diff),
        },
    )
    (mdir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"{ds_name}: exported {onnx_path.name} ({onnx_path.stat().st_size / 1e6:.2f} MB), max |diff| = {diff:.2e}")
    return meta


def main() -> None:
    for ds_name in DATASETS:
        export(ds_name)


if __name__ == "__main__":
    main()
