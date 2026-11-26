"""Domain variance: the binary-F1 grid and one-way eta-squared per experimental factor.

Slice: FRAUD-ICL few-shot (new format, Definitions 0-3, cosine retrieval, all four orderings),
valid responses only. A cell is model x ordering x definition x domain (4 x 4 x 4 x 7 = 448);
its score is deceptive-class binary F1, rounded to four decimals. Eta-squared is one-way, computed
separately for each factor over the 448 cell F1s; it measures association, not a variance
partition, and the four values need not sum to one.

    uv run python Analysis/domain_variance.py

Outputs (Outputs/analysis/domain_variance/):
    f1_grid.csv        448 cells: model, order, definition, domain, f1, n_valid, n_deceptive
    eta_squared.csv    one-way eta-squared for domain, model, definition and order
    domain_means.csv   mean, sd, min and max cell F1 per domain, ranked
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from common import ANALYSIS_DIR, DEFS, DOMAINS, MODELS, ORDERS, base_slice, f1_binary, load_model, parser, write_csv

FACTORS = ["domain", "model", "definition", "order"]


def eta_squared(df: pd.DataFrame, factor: str, value: str = "f1") -> float:
    """Between-group sum of squares over total sum of squares for one factor."""
    grand = df[value].mean()
    ss_total = ((df[value] - grand) ** 2).sum()
    if ss_total == 0:
        return 0.0
    ss_between = df.groupby(factor)[value].apply(lambda g: len(g) * (g.mean() - grand) ** 2).sum()
    return float(ss_between / ss_total)


def f1_grid(run_dir, compiled_dir) -> pd.DataFrame:
    cells = []
    for m in MODELS:
        v = base_slice(load_model(m, compiled_dir, run_dir))
        v = v[v.ok]
        for o in ORDERS:
            for k in DEFS:
                c = v[(v.order == o) & (v.definition == k)]
                for dom in DOMAINS:
                    g = c[c.domain == dom]
                    if len(g) == 0:
                        raise SystemExit(f"empty cell {m}/{o}/{k}/{dom}")
                    cells.append({"model": m, "order": o, "definition": k, "domain": dom,
                                  "f1": round(f1_binary(g.t, g.p), 4), "n_valid": len(g),
                                  "n_deceptive": int((g.t == 1).sum())})
    grid = pd.DataFrame(cells)
    if len(grid) != 448:
        raise SystemExit(f"expected 448 cells, got {len(grid)}")
    return grid


def main() -> None:
    ap = parser(__doc__)
    ap.add_argument("--out", type=Path, default=ANALYSIS_DIR / "domain_variance")
    a = ap.parse_args()

    grid = f1_grid(a.run_dir, a.compiled_dir)
    eta = pd.DataFrame([{"factor": f, "eta_squared": round(eta_squared(grid, f), 4)} for f in FACTORS])
    means = (grid.groupby("domain")["f1"].agg(mean_f1="mean", sd_f1="std", min_f1="min", max_f1="max", n_cells="size")
             .reindex(DOMAINS).reset_index())
    means["rank"] = means["mean_f1"].rank(ascending=False, method="min").astype(int)
    means = means.sort_values("rank").reset_index(drop=True)

    write_csv(grid, a.out / "f1_grid.csv")
    write_csv(eta, a.out / "eta_squared.csv")
    write_csv(means, a.out / "domain_means.csv")
    print(eta.to_string(index=False))
    print(means.round(3).to_string(index=False))
    print(f"Wrote {a.out}")


if __name__ == "__main__":
    main()
