"""Part A — full results on the test split (brief Part A steps 2-5).

Reads the predictions and SHAP importances written by src/part_a/tune.py.

Outputs
  results/part_a_test_scores.csv     test ROC-AUC (primary) and AP (secondary), mean ± sd over seeds
  results/part_a_leakage_demo.csv    naive vs cross-fitted target encoding: train / val / test, with the gap
  results/part_a_seen_unseen.csv     ROC-AUC on rows whose highest-cardinality level is seen / rare / unseen
                                     in training, plus the all-unseen stress test
  results/part_a_shap_agreement.csv  CatBoost vs LightGBM+TE SHAP: Kendall tau, top-10 overlap
  results/tables/part_a_*.md         the same as Markdown
  report/figures/part_a_*.pdf|png    test scores, SHAP top-10 comparison

Run from repo root:  uv run python src/part_a/analysis.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from data.datasets import load_dataset
from data.splits import load_split
from src.eval.metrics import avg_precision, paired_bootstrap_diff, roc_auc
from src.eval.plots import MODEL_COLORS, MODEL_LABELS, plt, save, setup
from src.eval.slices import category_slices, highest_cardinality_feature

RES = ROOT / "results"
DATASETS = ["adult", "bank-marketing", "amazon_employee_access"]
MODELS = ["onehot_lr", "ordinal_hgb", "te_lgbm", "catboost"]
MIN_SLICE = 30


def safe_auc(y, p):
    return roc_auc(y, p) if len(np.unique(y)) == 2 else np.nan


def load_preds(ds_name: str, model: str) -> list[dict]:
    out = []
    for f in sorted((RES / "predictions" / ds_name).glob(f"{model}_seed*.npz")):
        with np.load(f) as z:
            out.append({k: z[k] for k in z.files})
    return out


def sd(x):
    return float(np.std(x, ddof=1)) if len(x) > 1 else np.nan


def md(df: pd.DataFrame, name: str) -> None:
    (RES / "tables").mkdir(exist_ok=True)
    (RES / "tables" / f"{name}.md").write_text(df.to_markdown(index=False, floatfmt=".4f"))


def main() -> None:
    setup()
    scores, leak, slices, agree, boot = [], [], [], [], []
    shap_tables = {}
    for ds_name in DATASETS:
        ds = load_dataset(ds_name)
        sp = load_split(ds_name)
        y = {k: ds.y[sp[k]] for k in ("train", "val", "test")}
        X_tr, X_te = ds.X.iloc[sp["train"]], ds.X.iloc[sp["test"]]
        hc = highest_cardinality_feature(X_tr, ds.cat_cols)
        sl = category_slices(X_tr, X_te, hc)
        preds = {m: load_preds(ds_name, m) for m in MODELS + ["naive_te_lgbm"]}

        # 1. test scores
        for m in MODELS:
            auc = [roc_auc(y["test"], p["test"]) for p in preds[m]]
            ap = [avg_precision(y["test"], p["test"]) for p in preds[m]]
            val = [roc_auc(y["val"], p["val"]) for p in preds[m]]
            scores.append(dict(dataset=ds_name, model=m, n_seeds=len(auc), val_auc=np.mean(val),
                               test_auc=np.mean(auc), test_auc_sd=sd(auc), test_ap=np.mean(ap), test_ap_sd=sd(ap)))

        # best encoding vs CatBoost: paired bootstrap on the test rows (seed-averaged predictions)
        mean = lambda m, s="test": np.mean([p[s] for p in preds[m]], axis=0)
        for other in ("te_lgbm", "onehot_lr", "ordinal_hgb"):
            r = paired_bootstrap_diff(y["test"], mean("catboost"), mean(other), n_resamples=2000, seed=2026)
            boot.append(dict(dataset=ds_name, comparison=f"catboost - {other}", diff_auc=r["mean_diff"],
                             ci_low=r["ci_low"], ci_high=r["ci_high"], p_catboost_better=r["p_greater"]))

        # 2. leakage demonstration
        for m in ("te_lgbm", "naive_te_lgbm"):
            row = dict(dataset=ds_name, model=m)
            for s in ("train", "val", "test"):
                row[f"{s}_auc"] = np.mean([roc_auc(y[s], p[s]) for p in preds[m]])
            row["train_minus_test"] = row["train_auc"] - row["test_auc"]
            row["val_minus_test"] = row["val_auc"] - row["test_auc"]
            leak.append(row)

        # 4. rare and unseen categories of the highest-cardinality feature
        for m in MODELS:
            p = mean(m)
            row = dict(dataset=ds_name, feature=hc, model=m, auc_all=safe_auc(y["test"], p))
            for s in ("seen", "rare", "unseen"):
                k = sl == s
                row[f"n_{s}"] = int(k.sum())
                row[f"auc_{s}"] = safe_auc(y["test"][k], p[k]) if k.sum() >= MIN_SLICE else np.nan
            row["auc_stress_all_unseen"] = safe_auc(y["test"], mean(m, "test_unseen"))
            row["stress_drop"] = row["auc_all"] - row["auc_stress_all_unseen"]
            slices.append(row)

        # 5. SHAP agreement CatBoost vs LightGBM + target encoding
        shap = pd.read_csv(RES / f"m4_shap_importance_{ds_name}.csv")
        cb = shap[shap.model == "catboost"].set_index("feature").mean_abs_shap
        lg = shap[shap.model == "te_lgbm"].set_index("feature").mean_abs_shap.reindex(cb.index)
        k = min(10, len(cb))
        top_cb = list(cb.sort_values(ascending=False).index[:k])
        top_lg = list(lg.sort_values(ascending=False).index[:k])
        tau, p_tau = kendalltau(cb, lg)
        agree.append(dict(dataset=ds_name, n_features=len(cb), kendall_tau_all=tau, p_value=p_tau,
                          top10_overlap=len(set(top_cb) & set(top_lg)) / k,
                          top5_overlap=len(set(top_cb[:5]) & set(top_lg[:5])) / 5,
                          same_top1=top_cb[0] == top_lg[0],
                          catboost_top5=", ".join(top_cb[:5]), lgbm_te_top5=", ".join(top_lg[:5])))
        shap_tables[ds_name] = pd.DataFrame({"CatBoost": cb, "LightGBM + TE": lg})

    out = {"part_a_test_scores": scores, "part_a_bootstrap": boot, "part_a_leakage_demo": leak,
           "part_a_seen_unseen": slices, "part_a_shap_agreement": agree}
    for name, rows in out.items():
        df = pd.DataFrame(rows)
        df.to_csv(RES / f"{name}.csv", index=False)
        md(df, name)
        print(f"\n== {name}\n{df.round(4).to_string()}")
    plot_scores(pd.DataFrame(scores))
    plot_shap(shap_tables)
    local_explanations()


def plot_scores(s: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(10, 2.4), squeeze=False)
    for ax, ds in zip(axes[0], DATASETS):
        d = s[s.dataset == ds].set_index("model")
        for i, m in enumerate(MODELS):
            r = d.loc[m]
            ax.errorbar(r.test_auc, i, xerr=0 if np.isnan(r.test_auc_sd) else r.test_auc_sd, fmt="o",
                        color=MODEL_COLORS[m], ms=6, mec="white", mew=1.5, capsize=3, lw=1.5)
        ax.set_yticks(range(len(MODELS)), [MODEL_LABELS[m] for m in MODELS] if ax is axes[0][0] else [""] * 4)
        ax.set_ylim(-0.6, len(MODELS) - 0.4)
        ax.set_title(ds)
        ax.set_xlabel("test ROC-AUC")
    save(fig, "part_a_test_scores")


def plot_shap(tables: dict[str, pd.DataFrame]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), squeeze=False)
    colors = [MODEL_COLORS["catboost"], MODEL_COLORS["te_lgbm"]]
    for ax, (ds, t) in zip(axes[0], tables.items()):
        t = (t / t.sum()).sort_values("CatBoost").tail(10)
        n = len(t)
        for j, (c, col) in enumerate(zip(t.columns, colors)):
            ax.barh(np.arange(n) + (j - 0.5) * 0.4, t[c], height=0.36, color=col, label=c)
        ax.set_yticks(range(n), t.index, fontsize=7)
        ax.set_title(ds)
        ax.set_xlabel("share of mean |SHAP|")
        ax.grid(axis="y", visible=False)
    axes[0][0].legend(loc="lower right", fontsize=7)
    save(fig, "part_a_shap_top10")


def local_explanations(ds_name: str = "amazon_employee_access") -> None:
    """Local CatBoost SHAP for two test rows: one whose RESOURCE is seen (>= 10 training rows)
    and one whose RESOURCE never occurs in training. Saved model = seed 0 of the tuned config."""
    from catboost import CatBoostClassifier, Pool

    from src.part_a.pipelines import to_catboost_frame

    ds = load_dataset(ds_name)
    sp = load_split(ds_name)
    X_tr, X_te = ds.X.iloc[sp["train"]], ds.X.iloc[sp["test"]]
    hc = highest_cardinality_feature(X_tr, ds.cat_cols)
    sl = category_slices(X_tr, X_te, hc)
    model = CatBoostClassifier()
    model.load_model(str(ROOT / "models" / ds_name / "catboost.cbm"))
    cols = ds.num_cols + ds.cat_cols
    rows = {"seen RESOURCE": int(np.flatnonzero(sl == "seen")[0]), "unseen RESOURCE": int(np.flatnonzero(sl == "unseen")[0])}
    fig, axes = plt.subplots(1, 2, figsize=(9, 3), sharex=True)
    out = []
    for ax, (label, i) in zip(axes, rows.items()):
        x = to_catboost_frame(X_te.iloc[[i]][cols], ds.cat_cols)
        sv = model.get_feature_importance(Pool(x, cat_features=list(range(len(cols)))), type="ShapValues")[0]
        contrib = pd.Series(sv[:-1], index=cols).sort_values()
        p = float(model.predict_proba(x)[0, 1])
        ax.barh(contrib.index, contrib.values, color=[MODEL_COLORS["onehot_lr"] if v > 0 else MODEL_COLORS["ft_transformer"]
                                                       for v in contrib.values], height=0.6)
        ax.axvline(0, color="#52514e", lw=0.8)
        ax.set_title(f"{label}: p(access) = {p:.2f}, y = {ds.y[sp['test'][i]]}")
        ax.set_xlabel("SHAP (log-odds)")
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", labelsize=7)
        out += [dict(row=label, test_position=i, feature=f, shap=v, base_value=sv[-1], p=p) for f, v in contrib.items()]
    save(fig, "part_a_local_shap_amazon")
    pd.DataFrame(out).to_csv(RES / "part_a_local_shap_amazon.csv", index=False)


if __name__ == "__main__":
    main()
