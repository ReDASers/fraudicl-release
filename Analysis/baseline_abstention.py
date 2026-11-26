"""Headline prompting comparison with abstentions counted (the paper's baseline-abstention table).

Three conditions on the 3,000 evaluation queries:

    zero_shot   format new, Definition -1, no retrieval
    raddicl     format old, Definition 1, cosine, in-order
    fraudicl    format new, Definition 1, cosine, in-order

Each contrast (FRAUD-ICL - zero-shot, FRAUD-ICL - RADDICL) is scored two ways:

    valid        queries valid under both conditions (the paired deltas of the significance tables)
    abstain_err  all 3,000 queries, every invalid response counted as wrong (pred = 1 - gold):
                 a false negative on a deceptive query, a false positive on a non-deceptive one

F1 is tested by paired bootstrap over queries (10,000 resamples, seed 42), accuracy by McNemar with
Edwards' correction; Holm-Bonferroni within each slice x metric over 8 tests (2 contrasts x 4 models).

    uv run python Analysis/baseline_abstention.py

Outputs (Outputs/analysis/baseline_abstention/):
    arm_scores.csv               n, coverage, F1, accuracy and confusion counts per model x arm x slice
    contrasts.csv                deltas, bootstrap CIs, raw and Holm-adjusted p-values
    tab_baseline_abstention.tex  the table rows as printed in the paper
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from common import (ANALYSIS_DIR, DISPLAY, MODELS, N_EXAMPLES, confusion, f1_acc, holm, load_model, mcnemar,
                    paired_f1, parser, write_csv)

ARMS = {"zero_shot": ("new", -1, "none", "none"),
        "raddicl": ("old", 1, "cosine", "in_order"),
        "fraudicl": ("new", 1, "cosine", "in_order")}
CONTRASTS = [("fraudicl", "zero_shot"), ("fraudicl", "raddicl")]
SLICES = ["valid", "abstain_err"]


def arms_for(model: str, run_dir: Path, compiled_dir: Path) -> dict[str, pd.DataFrame]:
    d = load_model(model, compiled_dir, run_dir)
    out = {}
    for arm, (fmt, k, ret, order) in ARMS.items():
        g = d[(d.format == fmt) & (d.definition == k) & (d.retrieval == ret) & (d.order == order)]
        g = g.sort_values("example_id").reset_index(drop=True)
        if len(g) != N_EXAMPLES or g.example_id.duplicated().any():
            raise SystemExit(f"{model} {arm}: {len(g)} rows")
        out[arm] = g
    ids = out["fraudicl"].example_id.values
    assert all((g.example_id.values == ids).all() for g in out.values())
    return out


def slice_arm(d: pd.DataFrame, sl: str) -> pd.DataFrame:
    if sl == "valid":
        return d
    out = d.copy()
    out["p"] = np.where(out.ok, out.p, 1 - out.t)
    out["ok"] = True
    return out


def build(run_dir: Path, compiled_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    scores, contrasts = [], []
    for model in MODELS:
        arms = arms_for(model, run_dir, compiled_dir)
        for sl in SLICES:
            s = {a: slice_arm(g, sl) for a, g in arms.items()}
            for a in ARMS:
                g = s[a][s[a].ok]
                c = confusion(g.t, g.p).sum(1)
                f1, acc = f1_acc(c)
                scores.append([model, a, sl, len(g), arms[a].ok.mean(), f1, acc, *c.astype(int)])
            for a, b in CONTRASTS:
                both = (s[a].ok & s[b].ok).values
                t, pa, pb = s[a].t.values[both], s[a].p.values[both], s[b].p.values[both]
                d_f1, lo, hi, p_f1 = paired_f1(t, pa, pb)
                d_acc, _, _, p_mc = mcnemar(t, pa, pb)
                contrasts.append([sl, "f1", model, f"{a}-{b}", int(both.sum()), d_f1, lo, hi, p_f1])
                contrasts.append([sl, "acc", model, f"{a}-{b}", int(both.sum()), d_acc, np.nan, np.nan, p_mc])
    scores = pd.DataFrame(scores, columns=["model", "arm", "slice", "n", "coverage", "f1", "acc",
                                           "tp", "fp", "fn", "tn"])
    contrasts = pd.DataFrame(contrasts, columns=["slice", "metric", "model", "contrast", "n_pairs",
                                                 "delta", "ci_lo", "ci_hi", "p"])
    contrasts["p_holm8"] = contrasts.groupby(["slice", "metric"])["p"].transform(lambda x: holm(x.values))
    contrasts["sig"] = contrasts["p_holm8"] < 0.05
    return scores, contrasts


def tex(scores: pd.DataFrame, contrasts: pd.DataFrame) -> str:
    def sc(m, a, sl, col):
        return scores[(scores.model == m) & (scores.arm == a) & (scores.slice == sl)][col].iloc[0]

    def dl(m, con, sl, met):
        r = contrasts[(contrasts.model == m) & (contrasts.contrast == con) & (contrasts.slice == sl)
                      & (contrasts.metric == met)].iloc[0]
        v = f"{r.delta:+.3f}".replace("0.", ".")
        return f"${v}^\\dagger$" if (sl == "abstain_err" and r.sig) else f"${v}$"

    rows = []
    for m in MODELS:
        cov = " & ".join(f"{100 * sc(m, a, 'valid', 'coverage'):.1f}" for a in ARMS)
        f1 = " & ".join(f"{sc(m, a, 'abstain_err', 'f1'):.3f}" for a in ARMS)
        d = " & ".join(dl(m, con, sl, met) for met in ["f1", "acc"]
                       for con in ["fraudicl-zero_shot", "fraudicl-raddicl"] for sl in SLICES)
        rows.append(f"{DISPLAY[m]} & {cov} & {f1} & {d} \\\\")
    return "\n".join(rows) + "\n"


def main() -> None:
    ap = parser(__doc__)
    ap.add_argument("--out", type=Path, default=ANALYSIS_DIR / "baseline_abstention")
    a = ap.parse_args()

    scores, contrasts = build(a.run_dir, a.compiled_dir)
    write_csv(scores, a.out / "arm_scores.csv")
    write_csv(contrasts, a.out / "contrasts.csv")
    (a.out / "tab_baseline_abstention.tex").write_bytes(tex(scores, contrasts).encode("utf-8"))
    print(tex(scores, contrasts))
    print(f"Wrote {a.out}")


if __name__ == "__main__":
    main()
