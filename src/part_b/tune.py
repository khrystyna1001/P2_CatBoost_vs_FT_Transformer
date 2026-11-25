"""M4 — Part B: Optuna tuning, 5-seed final runs, test evaluation, attention + embeddings.

For every dataset and network (MLP with embeddings, FT-Transformer):
  1. Optuna TPE search with the SAME number of trials as Part A, objective =
     validation ROC-AUC (early stopping on validation ROC-AUC inside each trial,
     median pruning on the per-epoch validation AUC).
  2. Retrain the best configuration with 5 seeds; save validation / test
     predictions, plus the unseen-category stress test.
  3. FT-Transformer only: [CLS]-to-feature attention averaged over heads, layers
     and test rows (per seed), and the learned embeddings of the
     highest-cardinality feature (seed 0) projected with PCA and UMAP.
  4. Ablations (configurable): mean pooling instead of [CLS]; no rare-category
     bucketing (min_count = 1).

Run from repo root:  uv run python src/part_b/tune.py [--config configs/m4_part_b_tuning.yaml]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from data.datasets import load_dataset
from src.eval.metrics import roc_auc
from src.eval.slices import UNSEEN_TOKEN, highest_cardinality_feature
from src.part_b.models import count_parameters
from src.part_b.train import Tensors, predict, prepare, train

optuna.logging.set_verbosity(optuna.logging.WARNING)


FTT_SEARCH = {"d": [64, 96, 128, 192], "n_blocks": [1, 3]}


def search_space(trial: optuna.Trial, kind: str) -> dict:
    common = dict(lr=trial.suggest_float("lr", 1e-5, 3e-3, log=True),
                  weight_decay=trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
                  batch_size=512)
    if kind == "mlp":
        return dict(d_hidden=trial.suggest_categorical("d_hidden", [64, 128, 256, 512]),
                    n_layers=trial.suggest_int("n_layers", 1, 4),
                    dropout=trial.suggest_float("dropout", 0.0, 0.5), **common)
    # FT-Transformer: ranges from the M1 proposal, narrowed for CPU via the config (ftt_search)
    return dict(d=trial.suggest_categorical("d", FTT_SEARCH["d"]),
                n_blocks=trial.suggest_int("n_blocks", min(FTT_SEARCH["n_blocks"]), max(FTT_SEARCH["n_blocks"])),
                n_heads=8,
                attn_dropout=trial.suggest_float("attn_dropout", 0.0, 0.5),
                ffn_dropout=trial.suggest_float("ffn_dropout", 0.0, 0.5),
                residual_dropout=trial.suggest_float("residual_dropout", 0.0, 0.2), **common)


def stress_tensors(data, X_test: pd.DataFrame, feature: str) -> Tensors:
    X = X_test.copy()
    X[feature] = UNSEEN_TOKEN
    xn, xc = data.pre.transform(X)
    return Tensors(torch.from_numpy(xn), torch.from_numpy(xc), data.test.y)


@torch.no_grad()
def cls_attention(model, t: Tensors, batch_size: int = 1024) -> np.ndarray:
    """Mean attention from [CLS] (query 0) to each feature token, averaged over heads, layers and rows."""
    model.eval()
    tot, n = None, 0
    for i in range(0, len(t.y), batch_size):
        _, att = model(t.x_num[i:i + batch_size], t.x_cat[i:i + batch_size], return_attention=True)
        a = att[:, :, :, 0, 1:].mean(dim=(1, 2)).sum(0)  # (features,)
        tot = a if tot is None else tot + a
        n += att.shape[0]
    return (tot / n).numpy()


def embedding_projection(model, data, X_train: pd.DataFrame, y_train: np.ndarray, feature: str, seed: int) -> pd.DataFrame:
    from sklearn.decomposition import PCA

    j = data.pre.cat_cols.index(feature)
    E = model.tokenizer.category_embeddings(j, data.pre.cardinalities).detach().numpy()
    vocab = data.pre.vocabs[feature]
    inv = {i: v for v, i in vocab.items()}
    idx = data.pre.transform_cat(X_train)[:, j]
    stats = pd.DataFrame({"index": idx, "y": y_train}).groupby("index")["y"].agg(["mean", "count"])
    df = pd.DataFrame({"index": np.arange(len(E))})
    df["category"] = df["index"].map(inv).fillna("<UNK/rare>")
    df = df.join(stats, on="index").rename(columns={"mean": "train_target_mean", "count": "train_count"})
    pcs = PCA(n_components=2, random_state=seed).fit_transform(E)
    df["pc1"], df["pc2"] = pcs[:, 0], pcs[:, 1]
    try:
        import umap

        um = umap.UMAP(n_components=2, random_state=seed, n_neighbors=min(15, len(E) - 1)).fit_transform(E)
        df["umap1"], df["umap2"] = um[:, 0], um[:, 1]
    except Exception as exc:  # UMAP is optional
        print(f"UMAP skipped: {exc}")
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "m4_part_b_tuning.yaml"))
    ap.add_argument("--datasets", nargs="*")
    ap.add_argument("--models", nargs="*")
    ap.add_argument("--skip-tuning", action="store_true", help="reuse results/best_params/part_b_<ds>.json")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    FTT_SEARCH.update(cfg.get("ftt_search", {}))
    res = ROOT / "results"
    pred_dir = ROOT / cfg["predictions_dir"]

    for ds_name in args.datasets or cfg["datasets"]:
        ds = load_dataset(ds_name)
        data = prepare(ds, min_count=cfg["min_count"], include_test=True)
        sp = data.splits
        X_tr, X_te = ds.X.iloc[sp["train"]], ds.X.iloc[sp["test"]]
        y_tr = ds.y[sp["train"]]
        hc = highest_cardinality_feature(X_tr, ds.cat_cols)
        stress = stress_tensors(data, X_te, hc)
        (pred_dir / ds_name).mkdir(parents=True, exist_ok=True)
        bp_path = res / "best_params" / f"part_b_{ds_name}.json"
        bp_path.parent.mkdir(parents=True, exist_ok=True)
        best_all = json.loads(bp_path.read_text()) if bp_path.exists() else {}
        for kind in args.models or cfg["models"]:
            runs, attn_rows = [], []
            if not args.skip_tuning or kind not in best_all:
                t0 = time.perf_counter()

                def objective(trial):
                    params = search_space(trial, kind)

                    def on_epoch(epoch, val_auc):
                        trial.report(val_auc, epoch)
                        if trial.should_prune():
                            raise optuna.TrialPruned()

                    r = train(kind, params, data, seed=0, max_epochs=cfg["tuning"]["max_epochs"],
                              patience=cfg["tuning"]["patience"], on_epoch=on_epoch)
                    trial.set_user_attr("epochs", len(r.history))
                    return r.best_val_auc

                study = optuna.create_study(
                    direction="maximize", sampler=optuna.samplers.TPESampler(seed=cfg["optuna_seed"]),
                    pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3))
                study.optimize(objective, n_trials=cfg["n_trials"])
                best = search_space(optuna.trial.FixedTrial(study.best_params), kind)
                best_all[kind] = dict(params=best, best_val_auc=study.best_value,
                                      tuning_time_s=time.perf_counter() - t0, n_trials=len(study.trials))
                bp_path.write_text(json.dumps(best_all, indent=2))
                (res / "optuna").mkdir(exist_ok=True)
                pd.DataFrame([dict(trial=t.number, value=t.value, **t.params, **t.user_attrs) for t in study.trials]
                             ).to_csv(res / "optuna" / f"part_b_{ds_name}_{kind}.csv", index=False)
                print(f"{ds_name} {kind}: best val AUC {study.best_value:.4f} ({best_all[kind]['tuning_time_s']:.0f}s) {best}",
                      flush=True)

            variants = {kind: best_all[kind]["params"]}
            if kind == "ft_transformer":
                for ab in cfg.get("ablations", []):
                    variants[f"ft_transformer_{ab}"] = dict(best_all[kind]["params"], pooling="mean") if ab == "mean_pool" \
                        else dict(best_all[kind]["params"])
            for vname, params in variants.items():
                vdata = data
                if vname.endswith("no_rare_bucket"):
                    vdata = prepare(ds, min_count=1, include_test=True)
                vstress = stress if vdata is data else stress_tensors(vdata, X_te, hc)
                seeds = cfg["seeds"] if vname == kind else cfg["seeds"][:cfg["ablation_seeds"]]
                for seed in seeds:
                    r = train(kind, params, vdata, seed, max_epochs=cfg["final"]["max_epochs"],
                              patience=cfg["final"]["patience"])
                    preds = {s: predict(r.model, t) for s, t in
                             (("train", vdata.train), ("val", vdata.val), ("test", vdata.test), ("test_unseen", vstress))}
                    np.savez_compressed(pred_dir / ds_name / f"{vname}_seed{seed}.npz", **preds)
                    runs.append(dict(dataset=ds_name, model=vname, seed=seed, fit_time_s=round(r.train_time_s, 1),
                                     best_iteration=r.best_epoch, n_params=count_parameters(r.model),
                                     **{f"{s}_auc": roc_auc(t.y.numpy(), preds[s]) for s, t in
                                        (("train", vdata.train), ("val", vdata.val), ("test", vdata.test))}))
                    print(runs[-1], flush=True)
                    (res / "curves").mkdir(exist_ok=True)
                    r.history.assign(dataset=ds_name, model=vname, seed=seed).to_csv(
                        res / "curves" / f"{ds_name}_{vname}_seed{seed}.csv", index=False)
                    if vname == "ft_transformer":
                        a = cls_attention(r.model, vdata.test)
                        attn_rows += [dict(dataset=ds_name, seed=seed, feature=f, cls_attention=v)
                                      for f, v in zip(ds.num_cols + ds.cat_cols, a)]
                    if vname == kind and seed == seeds[0]:
                        mdir = ROOT / "models" / ds_name
                        mdir.mkdir(parents=True, exist_ok=True)
                        torch.save(r.model.state_dict(), mdir / f"{kind}.pt")
                        (mdir / f"{kind}.json").write_text(json.dumps(dict(params=params, n_num=len(ds.num_cols),
                                                                           cardinalities=vdata.pre.cardinalities)))
                        vdata.pre.save(mdir / "preprocessor.json")
                        if kind == "ft_transformer":
                            embedding_projection(r.model, vdata, X_tr, y_tr, hc, seed).assign(feature=hc).to_csv(
                                res / f"m4_embeddings_{ds_name}.csv", index=False)
                pd.DataFrame(runs).to_csv(res / f"m4_part_b_runs_{ds_name}_{kind}.csv", index=False)
            if attn_rows:
                pd.DataFrame(attn_rows).to_csv(res / f"m4_attention_importance_{ds_name}.csv", index=False)


if __name__ == "__main__":
    main()
