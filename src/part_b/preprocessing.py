"""Part B pre-processing (DL S1): fitted on training rows only.

* Numerical features: QuantileTransformer -> standard normal.
* Categorical features: per-feature vocabulary built from TRAINING rows.
  Index 0 is reserved for unseen and rare categories (< ``min_count`` training
  rows); frequent categories get indices 1..K.

The fitted state is serialisable to JSON so the API can reproduce it exactly
without scikit-learn pickles (see ``to_dict`` / ``from_dict``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

UNK = 0


@dataclass
class TabularPreprocessor:
    num_cols: list[str]
    cat_cols: list[str]
    min_count: int = 10
    n_quantiles: int = 1000
    vocabs: dict[str, dict[str, int]] = field(default_factory=dict)
    quantiles: dict[str, list[float]] = field(default_factory=dict)

    # ------------------------------------------------------------------ fit
    def fit(self, X: pd.DataFrame) -> "TabularPreprocessor":
        for col in self.cat_cols:
            counts = X[col].astype(str).value_counts()
            frequent = sorted(counts[counts >= self.min_count].index)
            self.vocabs[col] = {v: i + 1 for i, v in enumerate(frequent)}
        for col in self.num_cols:
            x = X[col].to_numpy(dtype=float)
            x = x[~np.isnan(x)]
            q = np.linspace(0, 1, min(self.n_quantiles, len(x)))
            self.quantiles[col] = np.quantile(x, q).tolist()
        return self

    # ------------------------------------------------------------ transform
    def transform_num(self, X: pd.DataFrame) -> np.ndarray:
        """Quantile -> uniform -> N(0,1), as sklearn's QuantileTransformer(output_distribution="normal")
        (interpolation averaged over the increasing and decreasing direction to handle ties).

        One deliberate difference: sklearn maps every value tied with the training minimum
        (maximum) to -5.2 (+5.2), which turns e.g. the 90% of zeros in capital-gain into
        extreme outliers. Here tied values map to the middle of their probability mass."""
        from scipy.special import ndtri

        out = np.zeros((len(X), len(self.num_cols)), dtype=np.float32)
        for j, col in enumerate(self.num_cols):
            ref = np.asarray(self.quantiles[col])
            refs = np.linspace(0, 1, len(ref))
            x = X[col].to_numpy(dtype=float)
            x = np.where(np.isnan(x), np.median(ref), x)
            u = 0.5 * (np.interp(x, ref, refs) - np.interp(-x, -ref[::-1], -refs[::-1]))
            eps = 1e-7
            out[:, j] = ndtri(np.clip(u, eps, 1 - eps))
        return out

    def transform_cat(self, X: pd.DataFrame) -> np.ndarray:
        out = np.zeros((len(X), len(self.cat_cols)), dtype=np.int64)
        for j, col in enumerate(self.cat_cols):
            vocab = self.vocabs[col]
            out[:, j] = X[col].astype(str).map(vocab).fillna(UNK).to_numpy(dtype=np.int64)
        return out

    def transform(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        return self.transform_num(X), self.transform_cat(X)

    @property
    def cardinalities(self) -> list[int]:
        """Embedding table sizes (K frequent levels + index 0 for unseen/rare)."""
        return [len(self.vocabs[c]) + 1 for c in self.cat_cols]

    # ------------------------------------------------------------ serialise
    def to_dict(self) -> dict:
        return dict(num_cols=self.num_cols, cat_cols=self.cat_cols, min_count=self.min_count,
                    n_quantiles=self.n_quantiles, vocabs=self.vocabs, quantiles=self.quantiles)

    @classmethod
    def from_dict(cls, d: dict) -> "TabularPreprocessor":
        return cls(**d)

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict()))

    @classmethod
    def load(cls, path: Path) -> "TabularPreprocessor":
        return cls.from_dict(json.loads(Path(path).read_text()))
