"""Part B models, written from scratch as nn.Module (DL S1-S3, S10-S12).

MLPWithEmbeddings
    One nn.Embedding per categorical feature, dimension min(50, ceil(levels/2)),
    concatenated with quantile-transformed numericals, then an MLP.

FTTransformer (Gorishniy et al., 2021)
    Feature tokenizer: numerical feature j -> x_j * W_j + b_j; categorical
    feature j -> E_j[c] + b_j. A learned [CLS] token is prepended and the
    sequence goes through pre-norm Transformer blocks. The prediction is made
    from the final [CLS] representation (or, for the ablation, from the mean
    of the feature tokens).

    The Transformer is nn.TransformerEncoder (pre-norm, 8 heads). Correctness is
    checked against rtdl_revisiting_models.FTTransformer in src/part_b/check_rtdl.py.
"""

from __future__ import annotations

import math

import torch
from torch import nn


def embedding_dim(levels: int) -> int:
    return int(min(50, math.ceil(levels / 2)))


class MLPWithEmbeddings(nn.Module):
    def __init__(self, n_num: int, cardinalities: list[int], d_hidden: int = 256, n_layers: int = 3,
                 dropout: float = 0.1):
        super().__init__()
        self.embeddings = nn.ModuleList(nn.Embedding(c, embedding_dim(c)) for c in cardinalities)
        d_in = n_num + sum(embedding_dim(c) for c in cardinalities)
        layers: list[nn.Module] = []
        for _ in range(n_layers):
            layers += [nn.Linear(d_in, d_hidden), nn.ReLU(), nn.Dropout(dropout)]
            d_in = d_hidden
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(d_in, 1)

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        parts = [x_num] + [emb(x_cat[:, j]) for j, emb in enumerate(self.embeddings)]
        return self.head(self.backbone(torch.cat(parts, dim=1))).squeeze(-1)


class FeatureTokenizer(nn.Module):
    def __init__(self, n_num: int, cardinalities: list[int], d: int):
        super().__init__()
        self.n_num = n_num
        self.num_weight = nn.Parameter(torch.empty(n_num, d))
        self.num_bias = nn.Parameter(torch.empty(n_num, d))
        offsets = torch.tensor([0] + list(cardinalities[:-1]), dtype=torch.long).cumsum(0)
        self.register_buffer("cat_offsets", offsets, persistent=False)
        self.cat_embeddings = nn.Embedding(max(sum(cardinalities), 1), d)
        self.cat_bias = nn.Parameter(torch.empty(len(cardinalities), d))
        s = 1 / math.sqrt(d)
        for p in (self.num_weight, self.num_bias, self.cat_bias, self.cat_embeddings.weight):
            nn.init.uniform_(p, -s, s)

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        tokens = []
        if self.n_num:
            tokens.append(x_num.unsqueeze(-1) * self.num_weight + self.num_bias)
        if x_cat.shape[1]:
            tokens.append(self.cat_embeddings(x_cat + self.cat_offsets) + self.cat_bias)
        return torch.cat(tokens, dim=1)

    def category_embeddings(self, feature_index: int, cardinalities: list[int]) -> torch.Tensor:
        start = int(sum(cardinalities[:feature_index]))
        return self.cat_embeddings.weight[start:start + cardinalities[feature_index]] + self.cat_bias[feature_index]


class ReGLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = x.chunk(2, dim=-1)
        return a * torch.relu(b)


class FTTransformer(nn.Module):
    """FT-Transformer on top of ``nn.TransformerEncoder`` (brief, Part B step 2).

    The feed-forward block of each nn.TransformerEncoderLayer is turned into the paper's ReGLU block:
    linear1: d -> 2*d_ffn, activation = ReGLU (halves the width), linear2: d_ffn -> d, with
    d_ffn = ffn_factor * d. The first layer keeps its pre-attention LayerNorm (rtdl skips it).
    Agreement with rtdl_revisiting_models.FTTransformer is checked in check_rtdl.py.
    """

    def __init__(self, n_num: int, cardinalities: list[int], d: int = 192, n_blocks: int = 3, n_heads: int = 8,
                 attn_dropout: float = 0.2, ffn_dropout: float = 0.1, residual_dropout: float = 0.0,
                 ffn_factor: float = 4 / 3, pooling: str = "cls"):
        super().__init__()
        assert pooling in {"cls", "mean"}
        self.cardinalities = list(cardinalities)
        self.pooling = pooling
        self.tokenizer = FeatureTokenizer(n_num, cardinalities, d)
        self.cls = nn.Parameter(torch.empty(1, 1, d))
        nn.init.uniform_(self.cls, -1 / math.sqrt(d), 1 / math.sqrt(d))
        d_ffn = int(d * ffn_factor)
        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=n_heads, dim_feedforward=2 * d_ffn, dropout=ffn_dropout,
            activation=ReGLU(), batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_blocks, enable_nested_tensor=False)
        for lyr in self.encoder.layers:
            lyr.linear2 = nn.Linear(d_ffn, d)  # ReGLU halves the hidden width
            lyr.self_attn.dropout = attn_dropout
            lyr.dropout1.p = lyr.dropout2.p = residual_dropout
        self.head = nn.Sequential(nn.LayerNorm(d), nn.ReLU(), nn.Linear(d, 1))

    def _tokens(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        tok = self.tokenizer(x_num, x_cat)
        return torch.cat([self.cls.expand(tok.shape[0], -1, -1), tok], dim=1)

    def _pool(self, x: torch.Tensor) -> torch.Tensor:
        pooled = x[:, 0] if self.pooling == "cls" else x[:, 1:].mean(dim=1)
        return self.head(pooled).squeeze(-1)

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor, return_attention: bool = False):
        x = self._tokens(x_num, x_cat)
        if not return_attention:
            return self._pool(self.encoder(x))
        # Same computation as nn.TransformerEncoder, layer by layer, to read the attention weights
        # (nn.TransformerEncoder itself does not return them). Equality with the plain path is tested.
        maps = []
        for lyr in self.encoder.layers:
            h = lyr.norm1(x)
            a, w = lyr.self_attn(h, h, h, need_weights=True, average_attn_weights=False)
            x = x + lyr.dropout1(a)
            x = x + lyr._ff_block(lyr.norm2(x))
            maps.append(w)  # (batch, heads, tokens, tokens)
        return self._pool(x), torch.stack(maps, dim=1)  # (batch, blocks, heads, tokens, tokens)


def build_model(kind: str, n_num: int, cardinalities: list[int], params: dict) -> nn.Module:
    params = {k: v for k, v in params.items() if k not in {"lr", "weight_decay", "batch_size"}}
    if kind == "mlp":
        return MLPWithEmbeddings(n_num, cardinalities, **params)
    if kind == "ft_transformer":
        return FTTransformer(n_num, cardinalities, **params)
    raise ValueError(kind)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
