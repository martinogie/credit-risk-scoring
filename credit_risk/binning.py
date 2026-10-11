"""Weight-of-Evidence (WoE) binning with monotonicity enforcement."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

EPS = 0.5  # Laplace smoothing for empty good/bad counts


@dataclass
class FeatureBins:
    name: str
    kind: str                       # "numeric" or "categorical"
    edges: list[float] = field(default_factory=list)   # numeric: inner cut points
    categories: list[str] = field(default_factory=list)
    woe: list[float] = field(default_factory=list)
    bad_rate: list[float] = field(default_factory=list)
    count: list[int] = field(default_factory=list)
    iv: float = 0.0

    def labels(self) -> list[str]:
        if self.kind == "categorical":
            return list(self.categories)
        e = [-np.inf] + list(self.edges) + [np.inf]
        out = []
        for lo, hi in zip(e[:-1], e[1:]):
            if np.isinf(lo):
                out.append(f"< {hi:g}")
            elif np.isinf(hi):
                out.append(f">= {lo:g}")
            else:
                out.append(f"[{lo:g}, {hi:g})")
        return out

    def bin_index(self, x: pd.Series) -> np.ndarray:
        if self.kind == "categorical":
            lookup = {c: i for i, c in enumerate(self.categories)}
            # Unseen categories map to the bin closest to the overall average (WoE nearest 0).
            fallback = int(np.argmin(np.abs(self.woe)))
            return x.astype(str).map(lookup).fillna(fallback).astype(int).to_numpy()
        return np.searchsorted(np.asarray(self.edges), x.to_numpy(dtype=float), side="right")

    def transform(self, x: pd.Series) -> np.ndarray:
        return np.asarray(self.woe)[self.bin_index(x)]


def _woe_table(idx: np.ndarray, y: np.ndarray, n_bins: int):
    total_bad, total_good = y.sum(), len(y) - y.sum()
    bad = np.bincount(idx, weights=y, minlength=n_bins)
    cnt = np.bincount(idx, minlength=n_bins)
    good = cnt - bad
    dist_bad = (bad + EPS) / (total_bad + EPS * n_bins)
    dist_good = (good + EPS) / (total_good + EPS * n_bins)
    woe = np.log(dist_good / dist_bad)          # positive WoE = safer than average
    iv = float(((dist_good - dist_bad) * woe).sum())
    bad_rate = np.divide(bad, cnt, out=np.zeros(n_bins), where=cnt > 0)
    return woe, iv, bad_rate, cnt


def fit_numeric(name: str, x: pd.Series, y: np.ndarray, direction: int,
                max_bins: int = 8, min_bin_frac: float = 0.05) -> FeatureBins:
    """Quantile pre-binning, then merge adjacent bins until bad rate is monotone in `direction`."""
    qs = np.unique(np.quantile(x, np.linspace(0, 1, max_bins + 1)[1:-1]))
    edges = list(qs)

    def stats(ed):
        idx = np.searchsorted(np.asarray(ed), x.to_numpy(dtype=float), side="right")
        return _woe_table(idx, y, len(ed) + 1)

    # Merge undersized bins.
    while edges:
        _, _, _, cnt = stats(edges)
        small = np.where(cnt < min_bin_frac * len(x))[0]
        if not len(small):
            break
        i = int(small[0])
        edges.pop(min(i, len(edges) - 1) if i < len(edges) else len(edges) - 1)

    # Enforce monotone bad rate (direction +1: rising with x; -1: falling; 0: leave as is).
    if direction != 0:
        while edges:
            _, _, br, _ = stats(edges)
            diffs = np.diff(br) * direction
            bad_steps = np.where(diffs < 0)[0]
            if not len(bad_steps):
                break
            edges.pop(int(bad_steps[0]))  # merge the violating pair

    woe, iv, br, cnt = stats(edges)
    return FeatureBins(name, "numeric", edges=[float(e) for e in edges], woe=woe.tolist(),
                       bad_rate=br.tolist(), count=cnt.astype(int).tolist(), iv=iv)


def fit_categorical(name: str, x: pd.Series, y: np.ndarray) -> FeatureBins:
    cats = sorted(x.astype(str).unique())
    lookup = {c: i for i, c in enumerate(cats)}
    idx = x.astype(str).map(lookup).to_numpy()
    woe, iv, br, cnt = _woe_table(idx, y, len(cats))
    return FeatureBins(name, "categorical", categories=cats, woe=woe.tolist(),
                       bad_rate=br.tolist(), count=cnt.astype(int).tolist(), iv=iv)
