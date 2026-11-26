"""Shared loading and metric helpers for the analysis scripts (CPU only).

The analysis scripts read the generation CSVs that `Inference/run_experiment.sh` writes to
`Outputs/full-run/<Model>-Generation/`. `compile_model` concatenates one model's CSVs, applies
the parse-error recovery in `fraudicl.recovery`, adds a stable per-query `example_id`, and caches
the result as `Outputs/compiled/<Model>_data.parquet`. Later calls read the cache unless a
generation CSV is newer than it.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from fraudicl.recovery import recover_frame  # noqa: E402

RUN_DIR = ROOT / "Outputs" / "full-run"
COMPILED_DIR = ROOT / "Outputs" / "compiled"
ANALYSIS_DIR = ROOT / "Outputs" / "analysis"
SAMPLES = ROOT / "Outputs" / "total_samples.csv"

MODELS = ["Qwen", "Gemma", "Mistral", "Neural"]
DISPLAY = {"Qwen": "Qwen", "Gemma": "Gemma", "Mistral": "Mistral", "Neural": "NeuralChat"}
ORDERS = ["in_order", "reverse_order", "random_order", "demo_order"]
DEFS = [0, 1, 2, 3]
DOMAINS = ["email", "message boards", "news", "political statements", "product reviews", "sms", "twitter"]
N_EXAMPLES = 3_000
DECIMALS = 4
COLS = ["definition", "format", "retrieval", "order", "domain", "true_label", "pred_label",
        "valid", "example_id", "sample_run"]


# -- loading ----------------------------------------------------------------------------------
def example_ids(df: pd.DataFrame) -> pd.Series:
    """Domain-qualified hash of the query text, so identical texts in two domains stay distinct."""
    key = df["domain"].astype(str) + "||" + df["prompt"].astype(str)
    return key.map(lambda s: hashlib.md5(s.encode("utf-8", errors="replace")).hexdigest()[:16])


def compile_model(model: str, run_dir: Path = RUN_DIR, compiled_dir: Path = COMPILED_DIR,
                  force: bool = False) -> Path:
    """Build (or reuse) the compiled parquet for one model and return its path."""
    sources = sorted((run_dir / f"{model}-Generation").glob("def_*_outputs.csv"))
    if not sources:
        raise SystemExit(f"no generation CSVs in {run_dir / f'{model}-Generation'}; "
                         "run Inference/run_experiment.sh first")
    out = compiled_dir / f"{model}_data.parquet"
    if out.exists() and not force and out.stat().st_mtime >= max(p.stat().st_mtime for p in sources):
        return out

    frames = []
    for path in sources:
        d = pd.read_csv(path, low_memory=False)
        d["source_file"] = path.name
        frames.append(d)
    d = pd.concat(frames, ignore_index=True)
    _, recovered, _ = recover_frame(d)

    ts = pd.read_csv(SAMPLES)
    sample_of = dict(zip(ts["domain"].astype(str) + "||" + ts["content"].astype(str),
                         "sample-" + ts["total_samples"].astype(str) + "-experiment"))
    d["sample_run"] = (d["domain"].astype(str) + "||" + d["prompt"].astype(str)).map(sample_of)
    if d["sample_run"].isna().any():
        raise SystemExit(f"{model}: {int(d['sample_run'].isna().sum())} rows are not in {SAMPLES.name}")
    d["example_id"] = example_ids(d)
    d["valid"] = d["valid"].astype(str).str.lower().eq("true")
    if "grounded" in d.columns:
        d["grounded"] = pd.to_numeric(d["grounded"], errors="coerce")

    compiled_dir.mkdir(parents=True, exist_ok=True)
    d.to_parquet(out, index=False)
    print(f"{model}: {len(sources)} CSVs, {len(d)} rows, {recovered} labels recovered -> {out}")
    return out


def load_model(model: str, compiled_dir: Path = COMPILED_DIR, run_dir: Path = RUN_DIR) -> pd.DataFrame:
    """One model's rows with convenience columns ok (valid), t (gold) and p (prediction, -1 if invalid)."""
    if run_dir.exists():
        path = compile_model(model, run_dir, compiled_dir)
    else:
        path = compiled_dir / f"{model}_data.parquet"
        if not path.exists():
            raise SystemExit(f"neither {run_dir} nor {path} exists; run Inference/run_experiment.sh first")
    d = pd.read_parquet(path, columns=COLS)
    if d["example_id"].nunique() != N_EXAMPLES:
        raise SystemExit(f"{model}: expected {N_EXAMPLES} queries, got {d['example_id'].nunique()}")
    d["model"] = model
    d["ok"] = d["valid"].astype(bool)
    d["t"] = d["true_label"].astype(int)
    d["p"] = pd.to_numeric(d["pred_label"], errors="coerce").fillna(-1).astype(int)
    if ((d["p"] == -1) != ~d["ok"]).any():
        raise SystemExit(f"{model}: valid flag and pred_label == -1 disagree")
    return d


def base_slice(d: pd.DataFrame) -> pd.DataFrame:
    """FRAUD-ICL few-shot: new format, Definitions 0-3, cosine retrieval, all four orders."""
    b = d[(d["format"] == "new") & d["definition"].isin(DEFS) & (d["retrieval"] == "cosine")]
    expected = len(DEFS) * len(ORDERS) * N_EXAMPLES
    if len(b) != expected or b.duplicated(["definition", "order", "example_id"]).any():
        raise SystemExit(f"{d['model'].iloc[0]}: base slice has {len(b)} rows (expected {expected}) "
                         "or duplicate keys")
    return b


# -- metrics ----------------------------------------------------------------------------------
def confusion(t, p) -> np.ndarray:
    """Shape (4, n): rows tp, fp, fn, tn, one column per query."""
    t, p = np.asarray(t), np.asarray(p)
    return np.stack([(t == 1) & (p == 1), (t == 0) & (p == 1),
                     (t == 1) & (p == 0), (t == 0) & (p == 0)]).astype(float)


def f1_acc(c) -> np.ndarray:
    """(F1, accuracy) from confusion counts tp, fp, fn, tn (vectorized)."""
    tp, fp, fn, tn = c
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = 2 * tp / (2 * tp + fp + fn)
        acc = (tp + tn) / (tp + fp + fn + tn)
    return np.stack([f1, acc])


def f1_binary(t, p) -> float:
    """Deceptive-class F1; 0.0 when there is no positive in truth or prediction."""
    tp, fp, fn, _ = confusion(t, p).sum(axis=1)
    den = 2 * tp + fp + fn
    return float(2 * tp / den) if den else 0.0


def holm(ps) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values."""
    ps = np.asarray(ps, float)
    out, running = np.empty(len(ps)), 0.0
    for rank, i in enumerate(np.argsort(ps, kind="stable")):
        running = max(running, min(1.0, (len(ps) - rank) * ps[i]))
        out[i] = running
    return out


def paired_f1(t, pa, pb, n_boot: int = 10_000, seed: int = 42, batch: int = 1_000):
    """Paired bootstrap over queries for F1(a) - F1(b): observed delta, 95% CI, two-sided p."""
    ca, cb = confusion(t, pa), confusion(t, pb)
    n = ca.shape[1]
    rng, parts, left = np.random.default_rng(seed), [], n_boot
    while left > 0:
        size = min(left, batch)
        idx = rng.integers(0, n, (size, n))
        parts.append(f1_acc(ca[:, idx].sum(2))[0] - f1_acc(cb[:, idx].sum(2))[0])
        left -= size
    ds = np.concatenate(parts)
    lo, hi = np.percentile(ds, [2.5, 97.5])
    obs = f1_acc(ca.sum(1)) - f1_acc(cb.sum(1))
    p = float(min(1.0, 2 * min((ds <= 0).mean(), (ds >= 0).mean())))
    return float(obs[0]), float(lo), float(hi), p


def mcnemar(t, pa, pb):
    """Accuracy difference and McNemar p-value with Edwards' continuity correction."""
    ca, cb = np.asarray(pa) == np.asarray(t), np.asarray(pb) == np.asarray(t)
    n1, n2 = int((ca & ~cb).sum()), int((~ca & cb).sum())
    if n1 + n2 == 0:
        return float(ca.mean() - cb.mean()), n1, n2, 1.0
    chi2 = (abs(n1 - n2) - 1) ** 2 / (n1 + n2)
    return float(ca.mean() - cb.mean()), n1, n2, float(st.chi2.sf(chi2, df=1))


# -- output -----------------------------------------------------------------------------------
def write_csv(df: pd.DataFrame, path: Path, decimals: int = DECIMALS) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(df.round(decimals).to_csv(index=False, lineterminator="\n").encode("utf-8"))


def parser(doc: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=doc.splitlines()[0])
    ap.add_argument("--run-dir", type=Path, default=RUN_DIR,
                    help="folder with the <Model>-Generation/ outputs (default: Outputs/full-run)")
    ap.add_argument("--compiled-dir", type=Path, default=COMPILED_DIR,
                    help="cache for the compiled per-model parquets (default: Outputs/compiled)")
    return ap
