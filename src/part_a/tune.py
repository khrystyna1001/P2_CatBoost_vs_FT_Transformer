"""M4 — Part A: Optuna tuning, final multi-seed fits, test evaluation, SHAP.

For every dataset and pipeline:
  1. Optuna TPE search, ``n_trials`` trials (equal budget for every model family,
     Part B uses the same number), objective = validation ROC-AUC.
  2. Refit the best configuration with 3 seeds (classical protocol) on the
     training split; save validation and test predictions, including a stress
     test where the highest-cardinality feature is replaced by an unseen level.
  3. SHAP: CatBoost (get_feature_importance(type="ShapValues")) and LightGBM with
     target encoding (TreeSHAP via pred_contrib) on the test split, seed 0.

The leaky ``naive_te_lgbm`` is not tuned separately: it reuses the tuned
LightGBM hyper-parameters of ``te_lgbm`` so the only difference is the encoder.

Run from repo root:  uv run python src/part_a/tune.py [--config configs/m4_part_a_tuning.yaml]
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
import yaml

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from catboost import Pool

from data.datasets import load_dataset
from data.splits import load_split
from src.eval.metrics import roc_auc
from src.eval.slices import UNSEEN_TOKEN, highest_cardinality_feature
from src.part_a.pipelines import RANDOMIZED, build_pipeline, fit_pipeline, predict_proba, to_catboost_frame

optuna.logging.set_verbosity(optuna.logging.WARNING)


def search_space(trial: optuna.Trial, name: str) -> dict:
    if name == "onehot_lr":
        return dict(C=trial.suggest_float("C", 1e-3, 1e2, log=True),
                    min_frequency=trial.suggest_int("min_frequency", 1, 50, log=True))
    if name == "ordinal_hgb":
        return dict(learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                    max_iter=trial.suggest_int("max_iter", 100, 1000, log=True),
                    max_leaf_nodes=trial.suggest_int("max_leaf_nodes", 8, 255, log=True),
                    min_samples_leaf=trial.suggest_int("min_samples_leaf", 5, 200, log=True),
                    l2_regularization=trial.suggest_float("l2_regularization", 1e-6, 10, log=True))
    if name == "te_lgbm":
        return dict(n_estimators=trial.suggest_int("n_estimators", 100, 1500, log=True),
                    learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                    num_leaves=trial.suggest_int("num_leaves", 8, 255, log=True),
                    min_child_samples=trial.suggest_int("min_child_samples", 5, 200, log=True),
                    reg_lambda=trial.suggest_float("reg_lambda", 1e-6, 10, log=True),
                    subsample=trial.suggest_float("subsample", 0.5, 1.0), subsample_freq=1,
                    colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
                    te_smooth=trial.suggest_float("te_smooth", 0.1, 100, log=True))
    if name == "catboost":  # ranges from the M1 proposal
        return dict(depth=trial.suggest_int("depth", 4, 10),
                    learning_rate=trial.suggest_float("learning_rate", 1e-2, 0.3, log=True),
                    l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1, 10),
                    random_strength=trial.suggest_float("random_strength", 1e-2, 10, log=True),
                    iterations=3000)
    raise ValueError(name)


def stress_unseen(X: pd.DataFrame, feature: str) -> pd.DataFrame:
    X = X.copy()
    X[feature] = UNSEEN_TOKEN
    return X


def shap_importance(name, model, X_te, cat, num) -> pd.Series:
    cols = num + cat
    if name == "catboost":
        pool = Pool(to_catboost_frame(X_te[cols], cat), cat_features=[cols.index(c) for c in cat])
        sv = model.get_feature_importance(pool, type="ShapValues")[:, :-1]
        return pd.Series(np.abs(sv).mean(0), index=cols)
    # te_lgbm: TreeSHAP on the encoded matrix; encoded columns are in [cat..., num...] order
    pre, clf = model.named_steps["preprocess"], model.named_steps["clf"]
    Z = pre.transform(X_te[cols])
    sv = clf.booster_.predict(Z, pred_contrib=True)[:, :-1]
    return pd.Series(np.abs(sv).mean(0), index=cat + num).reindex(cols)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "m4_part_a_tuning.yaml"))
    ap.add_argument("--datasets", nargs="*")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    pred_dir = ROOT / cfg["predictions_dir"]
    res_dir = ROOT / "results"

    for ds_name in args.datasets or cfg["datasets"]:
        ds = load_dataset(ds_name)
        sp = load_split(ds_name)
        X, y = ds.X, ds.y
        X_tr, X_va, X_te = (X.iloc[sp[k]] for k in ("train", "val", "test"))
        y_tr, y_va, y_te = (y[sp[k]] for k in ("train", "val", "test"))
        hc = highest_cardinality_feature(X_tr, ds.cat_cols)
        (pred_dir / ds_name).mkdir(parents=True, exist_ok=True)
        best_params_all, runs, shap_rows = {}, [], []

        for name in cfg["pipelines"]:
            t0 = time.perf_counter()

            def objective(trial):
                params = search_space(trial, name)
                m = build_pipeline(name, ds.cat_cols, ds.num_cols, params, seed=0)
                fr = fit_pipeline(name, m, X_tr, y_tr, X_va, y_va, ds.cat_cols, ds.num_cols)
                return roc_auc(y_va, predict_proba(name, fr.model, X_va, ds.cat_cols, ds.num_cols))

            study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=cfg["optuna_seed"]))
            study.optimize(objective, n_trials=cfg["n_trials"])
            tune_time = time.perf_counter() - t0
            best = dict(study.best_params)
            if name == "catboost":
                best["iterations"] = 3000
            if name == "te_lgbm":
                best["subsample_freq"] = 1
            best_params_all[name] = dict(params=best, best_val_auc=study.best_value, tuning_time_s=tune_time,
                                         n_trials=len(study.trials))
            print(f"{ds_name} {name}: best val AUC {study.best_value:.4f} after {len(study.trials)} trials "
                  f"({tune_time:.0f}s) {best}", flush=True)
            (res_dir / "optuna").mkdir(parents=True, exist_ok=True)
            pd.DataFrame([dict(trial=t.number, value=t.value, **t.params) for t in study.trials]).to_csv(
                res_dir / "optuna" / f"part_a_{ds_name}_{name}.csv", index=False)

        # te_lgbm hyper-parameters reused for the leaky pipeline (encoder is the only difference)
        lgbm_params = {k: v for k, v in best_params_all["te_lgbm"]["params"].items() if k != "te_smooth"}
        final = {n: best_params_all[n]["params"] for n in cfg["pipelines"]}
        final["naive_te_lgbm"] = lgbm_params

        for name, params in final.items():
            seeds = cfg["seeds"] if name in RANDOMIZED else cfg["seeds"][:1]
            for seed in seeds:
                m = build_pipeline(name, ds.cat_cols, ds.num_cols, dict(params), seed=seed)
                fr = fit_pipeline(name, m, X_tr, y_tr, X_va, y_va, ds.cat_cols, ds.num_cols)
                preds = {}
                for split, Xs in (("train", X_tr), ("val", X_va), ("test", X_te)):
                    preds[split] = predict_proba(name, fr.model, Xs, ds.cat_cols, ds.num_cols)
                preds["test_unseen"] = predict_proba(name, fr.model, stress_unseen(X_te, hc), ds.cat_cols, ds.num_cols)
                np.savez_compressed(pred_dir / ds_name / f"{name}_seed{seed}.npz", **preds)
                runs.append(dict(dataset=ds_name, model=name, seed=seed, fit_time_s=round(fr.fit_time_s, 2),
                                 best_iteration=fr.best_iteration,
                                 **{f"{s}_auc": roc_auc(yy, preds[s]) for s, yy in
                                    (("train", y_tr), ("val", y_va), ("test", y_te))}))
                print(runs[-1], flush=True)
                if seed == seeds[0] and name in ("catboost", "te_lgbm"):
                    imp = shap_importance(name, fr.model, X_te, ds.cat_cols, ds.num_cols)
                    shap_rows += [dict(dataset=ds_name, model=name, feature=f, mean_abs_shap=v) for f, v in imp.items()]
                if seed == seeds[0] and name == "catboost":
                    model_dir = ROOT / "models" / ds_name
                    model_dir.mkdir(parents=True, exist_ok=True)
                    fr.model.save_model(str(model_dir / "catboost.cbm"))

        (res_dir / "best_params").mkdir(exist_ok=True)
        (res_dir / "best_params" / f"part_a_{ds_name}.json").write_text(json.dumps(best_params_all, indent=2))
        pd.DataFrame(runs).to_csv(res_dir / f"m4_part_a_runs_{ds_name}.csv", index=False)
        pd.DataFrame(shap_rows).to_csv(res_dir / f"m4_shap_importance_{ds_name}.csv", index=False)


if __name__ == "__main__":
    main()
