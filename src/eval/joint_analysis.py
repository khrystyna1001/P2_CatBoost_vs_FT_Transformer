"""M4 — joint analysis of Part A and Part B on the locked test split.

Reads the saved predictions (results/predictions/<dataset>/<model>_seed<k>.npz)
and the importance files written by src/part_a/tune.py and src/part_b/tune.py.

Outputs (results/ and report/figures/):
  m4_test_scores.csv           test ROC-AUC / AP, mean ± sd over seeds, every model
  m4_bootstrap.csv             paired bootstrap CatBoost vs FT-Transformer (+ ROPE decision)
  m4_seen_unseen.csv           ROC-AUC on seen / rare / unseen slices of the highest-cardinality feature
                               and under the all-unseen stress test
  m4_leakage_demo.csv          naive vs cross-fitted target encoding, train / val / test
  m4_explanations.csv          Kendall tau: CatBoost SHAP vs FT-T attention, CatBoost vs LightGBM SHAP
  m4_ablations.csv             FT-Transformer [CLS] vs mean pooling vs no rare-category bucket
  tables/*.md                  the same tables as Markdown

Run from repo root:  uv run python src/eval/joint_analysis.py [--config configs/m4_joint.yaml]
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import kendalltau

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from data.datasets import load_dataset
from data.splits import load_split
from src.eval.metrics import avg_precision, roc_auc
from src.eval.plots import MODEL_COLORS, MODEL_LABELS, MUTED, PALETTE, plt, save, setup
from src.eval.slices import category_slices, highest_cardinality_feature

RES = ROOT / "results"


def safe_auc(y, p):
    return roc_auc(y, p) if len(np.unique(y)) == 2 else np.nan


def load_preds(ds_name: str) -> dict[str, dict[int, dict[str, np.ndarray]]]:
    out: dict = {}
    for f in sorted((RES / "predictions" / ds_name).glob("*_seed*.npz")):
        model, seed = f.stem.rsplit("_seed", 1)
        with np.load(f) as z:
            out.setdefault(model, {})[int(seed)] = {k: z[k] for k in z.files}
    return out


def mean_pred(preds: dict[int, dict], split: str) -> np.ndarray:
    return np.mean([p[split] for p in preds.values()], axis=0)


def bootstrap_seed_avg(y, preds_a, preds_b, metric_fn, n_resamples, seed, alpha=0.05):
    """Paired bootstrap of mean_s metric(model A, seed s) - mean_t metric(model B, seed t).

    Test rows are resampled with replacement and both families are scored on the same resample, so the
    comparison is paired. Averaging single-model scores (not scoring an average of the seeds' predictions)
    avoids giving the family with more seeds an ensemble advantage."""
    rng = np.random.default_rng(seed)
    n = len(y)
    diffs = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, n, n)
        yb = y[idx]
        diffs[i] = (np.mean([metric_fn(yb, p[idx]) for p in preds_a]) -
                    np.mean([metric_fn(yb, p[idx]) for p in preds_b]))
    lo, hi = np.quantile(diffs, [alpha / 2, 1 - alpha / 2])
    return dict(mean_diff=float(diffs.mean()), ci_low=float(lo), ci_high=float(hi), p_greater=float((diffs > 0).mean()))


def fmt(m, s):
    return f"{m:.4f} ± {s:.4f}" if np.isfinite(s) else f"{m:.4f}"


def to_md(df: pd.DataFrame, name: str) -> None:
    (RES / "tables").mkdir(exist_ok=True)
    (RES / "tables" / f"{name}.md").write_text(df.to_markdown(index=False, floatfmt=".4f"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "m4_joint.yaml"))
    cfg = yaml.safe_load(Path(ap.parse_args().config).read_text())
    setup()

    score_rows, boot_rows, slice_rows, leak_rows, expl_rows, abl_rows, rank_rows = [], [], [], [], [], [], []
    importance = {}

    for ds_name in cfg["datasets"]:
        ds = load_dataset(ds_name)
        sp = load_split(ds_name)
        y = {k: ds.y[sp[k]] for k in ("train", "val", "test")}
        X_tr, X_te = ds.X.iloc[sp["train"]], ds.X.iloc[sp["test"]]
        preds = load_preds(ds_name)

        # 1. test scores, mean ± sd over seeds
        for model, ps in preds.items():
            aucs = [roc_auc(y["test"], p["test"]) for p in ps.values()]
            aps = [avg_precision(y["test"], p["test"]) for p in ps.values()]
            vals = [roc_auc(y["val"], p["val"]) for p in ps.values()]
            score_rows.append(dict(dataset=ds_name, model=model, n_seeds=len(ps),
                                   val_auc_mean=np.mean(vals), test_auc_mean=np.mean(aucs),
                                   test_auc_sd=np.std(aucs, ddof=1) if len(aucs) > 1 else np.nan,
                                   test_ap_mean=np.mean(aps), test_ap_sd=np.std(aps, ddof=1) if len(aps) > 1 else np.nan))

        # 2. paired bootstrap CatBoost vs FT-Transformer
        a, b = preds["catboost"], preds["ft_transformer"]
        for metric_name, fn in (("roc_auc", roc_auc), ("avg_precision", avg_precision)):
            res = bootstrap_seed_avg(y["test"], [p["test"] for p in a.values()], [p["test"] for p in b.values()], fn,
                                     cfg["bootstrap_resamples"], cfg["bootstrap_seed"])
            rope = cfg["rope"][metric_name]
            if res["ci_low"] > -rope and res["ci_high"] < rope:
                decision = "practically equivalent"
            elif res["ci_low"] > rope:
                decision = "CatBoost better"
            elif res["ci_high"] < -rope:
                decision = "FT-Transformer better"
            else:
                decision = "undecided"
            # seed variation: every CatBoost seed vs every FT-T seed
            seed_diffs = [fn(y["test"], pa["test"]) - fn(y["test"], pb["test"])
                          for pa, pb in itertools.product(a.values(), b.values())]
            boot_rows.append(dict(dataset=ds_name, metric=metric_name, diff_catboost_minus_ftt=res["mean_diff"],
                                  ci_low=res["ci_low"], ci_high=res["ci_high"], p_catboost_better=res["p_greater"],
                                  rope=rope, decision=decision, seed_pair_diff_min=min(seed_diffs),
                                  seed_pair_diff_max=max(seed_diffs)))

        # 3. seen / rare / unseen slices + stress test
        hc = highest_cardinality_feature(X_tr, ds.cat_cols)
        sl = category_slices(X_tr, X_te, hc)
        for model in cfg["slice_models"]:
            seeds_p = list(preds[model].values())
            seed_mean = lambda key, m=slice(None): float(np.mean([safe_auc(y["test"][m], p[key][m]) for p in seeds_p]))
            row = dict(dataset=ds_name, feature=hc, model=model, all=seed_mean("test"))
            for s in ("seen", "rare", "unseen"):
                m = sl == s
                row[f"n_{s}"] = int(m.sum())
                row[f"auc_{s}"] = seed_mean("test", m) if m.sum() >= cfg["min_slice_rows"] else np.nan
            row["auc_stress_all_unseen"] = seed_mean("test_unseen")
            row["drop_stress"] = row["all"] - row["auc_stress_all_unseen"]
            slice_rows.append(row)

        # 4. leakage demonstration
        for model in ("te_lgbm", "naive_te_lgbm"):
            ps = preds[model]
            leak_rows.append(dict(dataset=ds_name, model=model, **{
                f"{s}_auc": np.mean([roc_auc(y[s], p[s]) for p in ps.values()]) for s in ("train", "val", "test")}))

        # 5. explanations
        shap = pd.read_csv(RES / f"m4_shap_importance_{ds_name}.csv")
        att = pd.read_csv(RES / f"m4_attention_importance_{ds_name}.csv")
        cb = shap[shap.model == "catboost"].set_index("feature").mean_abs_shap
        lg = shap[shap.model == "te_lgbm"].set_index("feature").mean_abs_shap.reindex(cb.index)
        at_seed = att.pivot(index="feature", columns="seed", values="cls_attention").reindex(cb.index)
        at = at_seed.mean(axis=1)
        importance[ds_name] = pd.DataFrame({"CatBoost |SHAP|": cb, "LightGBM+TE |SHAP|": lg, "FT-T [CLS] attention": at})
        k = min(10, len(cb))
        top = lambda s: set(s.sort_values(ascending=False).index[:k])
        stab = [kendalltau(at_seed[i], at_seed[j])[0] for i, j in itertools.combinations(at_seed.columns, 2)]
        rk = lambda ser: int(ser.rank(ascending=False)[hc])
        rank_rows.append(dict(dataset=ds_name, feature=hc, n_features=len(cb), n_levels=int(ds.X[hc].nunique()),
                              catboost_shap_rank=rk(cb), lightgbm_te_shap_rank=rk(lg), ftt_attention_rank=rk(at),
                              catboost_share=float(cb[hc] / cb.sum()), lightgbm_te_share=float(lg[hc] / lg.sum()),
                              ftt_attention_share=float(at[hc] / at.sum())))
        expl_rows.append(dict(
            dataset=ds_name, n_features=len(cb),
            tau_catboost_shap_vs_ftt_attention=kendalltau(cb, at)[0],
            p_catboost_shap_vs_ftt_attention=kendalltau(cb, at)[1],
            tau_catboost_shap_vs_lgbm_shap=kendalltau(cb, lg)[0],
            top10_overlap_catboost_lgbm=len(top(cb) & top(lg)) / k,
            top10_overlap_catboost_ftt=len(top(cb) & top(at)) / k,
            attention_seed_stability_tau=float(np.mean(stab)) if stab else np.nan,
            catboost_top3=", ".join(cb.sort_values(ascending=False).index[:3]),
            ftt_top3=", ".join(at.sort_values(ascending=False).index[:3])))

        # 6. ablations
        for model in ["ft_transformer"] + [f"ft_transformer_{a}" for a in cfg["ablations"]]:
            if model not in preds:
                continue
            ps = preds[model]
            aucs = [roc_auc(y["test"], p["test"]) for p in ps.values()]
            m = sl != "seen"
            abl_rows.append(dict(dataset=ds_name, variant=model, n_seeds=len(ps), test_auc_mean=np.mean(aucs),
                                 test_auc_sd=np.std(aucs, ddof=1) if len(aucs) > 1 else np.nan,
                                 auc_rare_or_unseen=float(np.mean([safe_auc(y["test"][m], p["test"][m]) for p in ps.values()]))
                                 if m.sum() >= cfg["min_slice_rows"] else np.nan,
                                 auc_stress_all_unseen=float(np.mean([safe_auc(y["test"], p["test_unseen"]) for p in ps.values()]))))

        plot_embeddings(ds_name)

    scores = pd.DataFrame(score_rows)
    scores.to_csv(RES / "m4_test_scores.csv", index=False)
    md = scores.assign(model=scores.model.map(lambda m: MODEL_LABELS.get(m, m)),
                       test_auc=[fmt(m, s) for m, s in zip(scores.test_auc_mean, scores.test_auc_sd)],
                       test_ap=[fmt(m, s) for m, s in zip(scores.test_ap_mean, scores.test_ap_sd)])
    to_md(md[["dataset", "model", "n_seeds", "val_auc_mean", "test_auc", "test_ap"]], "m4_test_scores")
    for name, rows in (("m4_bootstrap", boot_rows), ("m4_seen_unseen", slice_rows), ("m4_leakage_demo", leak_rows),
                       ("m4_explanations", expl_rows), ("m4_ablations", abl_rows),
                       ("m4_importance_rank_high_card_feature", rank_rows)):
        df = pd.DataFrame(rows)
        df.to_csv(RES / f"{name}.csv", index=False)
        to_md(df, name)
    plot_scores(scores, cfg)
    plot_importance(importance)
    print(scores.round(4).to_string())
    for name, rows in (("bootstrap", boot_rows), ("seen/unseen", slice_rows), ("leakage", leak_rows),
                       ("explanations", expl_rows), ("ablations", abl_rows), ("rank of high-cardinality feature", rank_rows)):
        print(f"\n== {name}\n{pd.DataFrame(rows).round(4).to_string()}")


def plot_scores(scores: pd.DataFrame, cfg: dict) -> None:
    order = ["onehot_lr", "ordinal_hgb", "te_lgbm", "catboost", "mlp", "ft_transformer"]
    fig, axes = plt.subplots(1, len(cfg["datasets"]), figsize=(10, 2.8), squeeze=False)
    axes = axes[0]
    for ax, ds in zip(axes, cfg["datasets"]):
        s = scores[scores.dataset == ds].set_index("model")
        for i, m in enumerate(order):
            if m not in s.index:
                continue
            r = s.loc[m]
            ax.errorbar(r.test_auc_mean, i, xerr=0 if np.isnan(r.test_auc_sd) else r.test_auc_sd, fmt="o",
                        color=MODEL_COLORS[m], ms=6, mec="white", mew=1.5, capsize=3, lw=1.5)
        ax.set_yticks(range(len(order)), [MODEL_LABELS[m] for m in order] if ax is axes[0] else [""] * len(order))
        ax.set_title(ds)
        ax.set_xlabel("test ROC-AUC")
        ax.axhline(3.5, color=MUTED, lw=0.6, ls=":")
    save(fig, "m4_test_scores")


def plot_importance(importance: dict[str, pd.DataFrame]) -> None:
    fig, axes = plt.subplots(1, len(importance), figsize=(11, 3.8), squeeze=False)
    axes = axes[0]
    cols = list(next(iter(importance.values())).columns)
    colors = [MODEL_COLORS["catboost"], MODEL_COLORS["te_lgbm"], MODEL_COLORS["ft_transformer"]]
    for ax, (ds, imp) in zip(axes, importance.items()):
        norm = imp / imp.sum()
        norm = norm.sort_values(cols[0])
        n = len(norm)
        h = 0.27
        for j, (c, col) in enumerate(zip(cols, colors)):
            ax.barh(np.arange(n) + (j - 1) * h, norm[c], height=h - 0.03, color=col, label=c)
        ax.set_yticks(range(n), norm.index, fontsize=7)
        ax.set_title(ds)
        ax.set_xlabel("share of total importance")
        ax.grid(axis="y", visible=False)
    axes[0].legend(loc="lower right", fontsize=7)
    save(fig, "m4_importance")


def plot_embeddings(ds_name: str) -> None:
    path = RES / f"m4_embeddings_{ds_name}.csv"
    if not path.exists():
        return
    e = pd.read_csv(path)
    proj = [("pc1", "pc2", "PCA")] + ([("umap1", "umap2", "UMAP")] if "umap1" in e else [])
    fig, axes = plt.subplots(1, len(proj), figsize=(4.2 * len(proj), 3.4), squeeze=False)
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("seq_blue", ["#d6e6f9", "#2a78d6", "#0d2f5c"])
    for ax, (a, b, title) in zip(axes[0], proj):
        unk = e["index"] == 0
        size = 6 + 40 * np.sqrt(e.train_count.fillna(0) / e.train_count.max())
        tm = e.loc[~unk, "train_target_mean"]
        vmin = float(max(0.0, np.floor(tm.quantile(0.02) * 20) / 20))  # colour range follows the data (Amazon: 94 % positives)
        sc = ax.scatter(e.loc[~unk, a], e.loc[~unk, b], c=tm, cmap=cmap, s=size[~unk], edgecolor="white",
                        linewidth=0.4, vmin=vmin, vmax=1)
        ax.scatter(e.loc[unk, a], e.loc[unk, b], marker="X", s=80, color=PALETTE[1], edgecolor="white", lw=1,
                   label="index 0 (unseen / rare)")
        ax.set_title(f"{title}")
        ax.set_xticks([]); ax.set_yticks([])
        ax.legend(loc="best", fontsize=7)
    fig.colorbar(sc, ax=axes[0].tolist(), label="training target mean", shrink=0.8)
    fig.suptitle(f"{ds_name}: FT-Transformer embeddings of '{e.feature.iloc[0]}' (marker size ~ training rows)", fontsize=10)
    save(fig, f"m4_embeddings_{ds_name}")


if __name__ == "__main__":
    main()
