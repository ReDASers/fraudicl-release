"""Verify the matched random-retrieval ablation produced by Scripts/run_random_retrieval.py.

Checks, per backbone (Outputs/random-retrieval/<Model>-Generation/def_1_new_random_outputs.csv):

  1. shape       500 rows, retrieval == random, order == in_order, definition == 1, format new,
                 one row per query in samples.csv, no duplicates.
  2. matched     every query has a cosine in_order partner row in Outputs/full-run, and in BOTH arms
                 re-rendering the logged demonstrations through today's PromptManager reproduces
                 raw_prompt byte for byte (same template; only the demonstrations differ). Both arms
                 must also use the same label -> tag wording ("non-deceptive"/"deceptive"); a
                 store using "not deceptive" fails here.
  3. demos       each row carries exactly 4 demonstrations, all of which are chunks of the
                 knowledge base (Saved/document_store; contents and order only), and the random set differs from the
                 cosine set for that query.
  4. seed        the 4 demonstrations are exactly what random.sample() yields after reseeding
                 with ExperimentManager._row_seed(...) for that row, i.e. the arm is
                 bitwise-reproducible (--per-row-seed). Requires loading the document store.

--compare additionally prints paired binary-F1 (random - cosine) on queries valid under both
conditions, with a 10,000-resample paired bootstrap CI clustered by query, plus the zero-shot
reference from the same full-run. Parse-error recovery (fraudicl.recovery) is applied to all
three arms identically.

Run from the repo root:  uv run python Scripts/verify_random_retrieval.py [--compare] [--no-seed-check]
Exit status is non-zero if any check fails.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

import fraudicl.global_config as global_config  # noqa: E402
from fraudicl.experimentmanager import ExperimentManager  # noqa: E402
from fraudicl.promptmanager import PromptManager  # noqa: E402
from fraudicl.recovery import recover_frame  # noqa: E402

MODELS = ["Neural", "Mistral", "Gemma", "Qwen"]
KEYS = {"Neural": "neural_chat", "Mistral": "mistral_instruct", "Gemma": "gemma", "Qwen": "qwen3"}
K = 4
N_EXPECTED = 500
RANDOM_DIR = _REPO_ROOT / "Outputs" / "random-retrieval"
COSINE_DIR = _REPO_ROOT / "Outputs" / "full-run"
CONTROL_DIR = _REPO_ROOT / "Outputs" / "random-retrieval-cosine-control"
STORE = _REPO_ROOT / "Saved" / "document_store"

COLS = ["definition", "format", "retrieval", "order", "prompt", "true_label", "pred_label",
        "pred_tag", "valid", "raw_answer", "raw_prompt", "retrieved_examples"]


def _valid(s: pd.Series) -> pd.Series:
    return s.astype(str).str.lower().eq("true")


def _read(path: Path, **kw) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False, **kw)
    df["definition"] = pd.to_numeric(df["definition"], errors="coerce")
    return df


def _demos(cell: str) -> list[dict]:
    try:
        return json.loads(cell)
    except (TypeError, ValueError):
        return []


def _render(pm: PromptManager, query: str, demos: list[dict], definition: int = 1) -> str:
    """The prompt today's code builds for this query from these logged demonstrations."""
    docs = [SimpleNamespace(content=d["content"], meta={k: d[k] for k in ("label", "tag", "domain")}) for d in demos]
    return pm.run(definition, False, docs, query.strip())["prompt"]


def _prompt_manager(model: str) -> PromptManager:
    return PromptManager(prompt_profile=global_config.LLM.get_model_profile(KEYS[model])["prompt_profile"])


def _tags(cells) -> set[tuple[int, str]]:
    return {(int(d["label"]), d["tag"]) for c in cells for d in _demos(c)}


def load_store_contents() -> list[str]:
    import h5py
    with h5py.File(STORE, "r") as f:
        return [c.decode("utf-8") for c in f["contents"][:]]


def check_model(model: str, samples: pd.DataFrame, store: list[str] | None, seed_check: bool) -> tuple[bool, pd.DataFrame | None]:
    ok = True
    def fail(msg):
        nonlocal ok
        ok = False
        print(f"  FAIL  {msg}")

    path = RANDOM_DIR / f"{model}-Generation" / "def_1_new_random_outputs.csv"
    if not path.exists():
        fail(f"missing {path.relative_to(_REPO_ROOT)}")
        return False, None
    r = _read(path, usecols=COLS)

    # 1. shape
    if len(r) != N_EXPECTED:
        fail(f"{len(r)} rows, expected {N_EXPECTED}")
    bad = r[(r.retrieval != "random") | (r.order != "in_order") | (r.definition != 1) | (r.format != "new")]
    if len(bad):
        fail(f"{len(bad)} rows outside the random/in_order/def1/new cell")
    if r.prompt.duplicated().any():
        fail(f"{r.prompt.duplicated().sum()} duplicate queries")
    missing = set(samples.content) - set(r.prompt)
    extra = set(r.prompt) - set(samples.content)
    if missing or extra:
        fail(f"query set mismatch vs samples.csv: {len(missing)} missing, {len(extra)} extra")

    # 2. cosine partner
    cpath = COSINE_DIR / f"{model}-Generation" / "def_1_new_cosine_outputs.csv"
    if not cpath.exists():
        fail(f"cosine partner missing: {cpath.relative_to(_REPO_ROOT)}")
        return ok, None
    c = _read(cpath, usecols=COLS)
    c = c[(c.order == "in_order")].drop_duplicates("prompt")
    j = r.merge(c, on="prompt", how="left", suffixes=("_rand", "_cos"))
    if j.raw_prompt_cos.isna().any():
        fail(f"{int(j.raw_prompt_cos.isna().sum())} queries have no cosine in_order partner")
    j = j[j.raw_prompt_cos.notna()]
    pm = _prompt_manager(model)
    n_bad_rand = sum(_render(pm, x.prompt, _demos(x.retrieved_examples_rand)) != x.raw_prompt_rand for x in j.itertuples())
    n_bad_cos = sum(_render(pm, x.prompt, _demos(x.retrieved_examples_cos)) != x.raw_prompt_cos for x in j.itertuples())
    if n_bad_rand or n_bad_cos:
        fail(f"prompts not reproduced by the current template: {n_bad_rand} random, {n_bad_cos} cosine rows")
    else:
        print(f"  ok    both arms' prompts reproduce byte for byte from one template ({len(j)} rows)")
    tr, tc = _tags(j.retrieved_examples_rand), _tags(j.retrieved_examples_cos)
    if tr != tc:
        fail(f"demonstration label wording differs between arms: random {sorted(tr)} vs cosine {sorted(tc)}")
    else:
        print(f"  ok    same demonstration label wording in both arms: {sorted(tr)}")

    # 3. demos
    store_set = set(s.strip() for s in store) if store is not None else None
    n_wrong_k = n_not_kb = n_same_as_cos = 0
    for _, row in j.iterrows():
        dr = _demos(row.retrieved_examples_rand)
        dc = _demos(row.retrieved_examples_cos)
        if len(dr) != K:
            n_wrong_k += 1
        if store_set is not None and any(d.get("content", "").strip() not in store_set for d in dr):
            n_not_kb += 1
        if [d.get("content") for d in dr] == [d.get("content") for d in dc]:
            n_same_as_cos += 1
    if n_wrong_k:
        fail(f"{n_wrong_k} rows without exactly {K} demonstrations")
    if n_not_kb:
        fail(f"{n_not_kb} rows with a demonstration not found in the knowledge base")
    if n_same_as_cos:
        fail(f"{n_same_as_cos} rows whose random set equals the cosine set")
    if not (n_wrong_k or n_not_kb or n_same_as_cos):
        print(f"  ok    every row has {K} KB demonstrations, none identical to its cosine set")

    # 4. seed reproducibility
    if seed_check and store is not None:
        # RetrievalManager.random() is random.sample(all_documents, k) over the store's
        # document order, after ExperimentManager reseeds from the row identity.
        # Rows are written in samples.csv order, so row index == position in samples.csv.
        pos = {content: i for i, content in enumerate(samples.content)}
        n_bad = 0
        for _, row in r.iterrows():
            idx = pos.get(row.prompt)
            if idx is None:
                continue
            random.seed(ExperimentManager._row_seed(global_config.General.RANDOM_SEED, idx, 1, "new", "random", "in_order"))
            expected = [s.strip() for s in random.sample(store, K)]
            got = [d.get("content", "").strip() for d in _demos(row.retrieved_examples)]
            if expected != got:
                n_bad += 1
        if n_bad:
            fail(f"{n_bad} rows whose demonstrations do not match the per-row-seed re-derivation")
        else:
            print(f"  ok    all {len(r)} rows reproduce from --per-row-seed")
    return ok, j


def compare(model: str, j: pd.DataFrame, rng: np.random.Generator) -> None:
    from sklearn.metrics import f1_score

    def arm(suffix):
        d = pd.DataFrame({
            "prompt": j.prompt, "true_label": j[f"true_label{suffix}"], "pred_label": j[f"pred_label{suffix}"],
            "pred_tag": j[f"pred_tag{suffix}"], "valid": j[f"valid{suffix}"], "raw_answer": j[f"raw_answer{suffix}"]})
        d, _, _ = recover_frame(d)
        d["pred_label"] = pd.to_numeric(d.pred_label, errors="coerce")
        d["true_label"] = pd.to_numeric(d.true_label, errors="coerce")
        d["v"] = _valid(d.valid)
        return d

    rand, cos = arm("_rand"), arm("_cos")
    zpath = COSINE_DIR / f"{model}-Generation" / "def_-1_new_none_outputs.csv"
    zs_line = ""
    if zpath.exists():
        try:
            z = _read(zpath, usecols=["definition", "prompt", "true_label", "pred_label", "pred_tag", "valid", "raw_answer", "raw_prompt"])
            z = z[z.prompt.isin(j.prompt)]
            pm = _prompt_manager(model)
            z = z[[_render(pm, x.prompt, [], -1) == x.raw_prompt for x in z.itertuples()]]
            z = z.drop_duplicates("prompt", keep="last")  # a resumed run may log a query twice
            z, _, _ = recover_frame(z)
            z["v"] = _valid(z.valid)
            zv = z[z.v]
            zs_line = f" | zero-shot F1 {f1_score(zv.true_label.astype(int), pd.to_numeric(zv.pred_label).astype(int)):.3f} (cov {z.v.mean():.3f}, n={len(z)})"
        except Exception as exc:  # report rather than abort on an unreadable file
            zs_line = f" | zero-shot unreadable: {exc.__class__.__name__}"

    both = rand.v.values & cos.v.values
    y = rand.true_label.values[both].astype(int)
    a = rand.pred_label.values[both].astype(int)
    b = cos.pred_label.values[both].astype(int)
    f_r, f_c = f1_score(y, a), f1_score(y, b)
    n = both.sum()
    bs = []
    for _ in range(10_000):
        w = rng.integers(0, n, n)
        bs.append(f1_score(y[w], a[w]) - f1_score(y[w], b[w]))
    lo, hi = np.percentile(bs, [2.5, 97.5])
    print(f"  F1 random {f_r:.3f} (cov {rand.v.mean():.3f}) | cosine {f_c:.3f} (cov {cos.v.mean():.3f}) | "
          f"random-cosine {f_r - f_c:+.3f} [{lo:+.3f}, {hi:+.3f}] n={n}{zs_line}")


def control(model: str) -> None:
    """How far does this machine alone move the output? Cosine rows regenerated locally vs full-run."""
    path = CONTROL_DIR / f"{model}-Generation" / "def_1_new_cosine_outputs.csv"
    if not path.exists():
        print("  control not run")
        return
    a = _read(path, usecols=COLS)
    b = _read(COSINE_DIR / f"{model}-Generation" / "def_1_new_cosine_outputs.csv", usecols=COLS)
    k = a.merge(b[b.order == "in_order"].drop_duplicates("prompt"), on="prompt", suffixes=("_loc", "_full"))
    same = lambda c: (k[f"{c}_loc"].astype(str) == k[f"{c}_full"].astype(str)).mean()
    lab = (pd.to_numeric(k.pred_label_loc, errors="coerce") == pd.to_numeric(k.pred_label_full, errors="coerce")).mean()
    print(f"  cosine control (this machine vs full-run, n={len(k)}): same prompt {same('raw_prompt'):.3f}, "
          f"identical answer {same('raw_answer'):.3f}, same label {lab:.3f}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--compare", action="store_true", help="also print paired F1 random vs cosine")
    ap.add_argument("--no-seed-check", action="store_true", help="skip the per-row-seed re-derivation (no store load)")
    ap.add_argument("--models", nargs="+", default=MODELS)
    args = ap.parse_args()

    samples_path = RANDOM_DIR / "samples.csv"
    if not samples_path.exists():
        print(f"missing {samples_path}"); return 1
    samples = pd.read_csv(samples_path)
    store = None if args.no_seed_check else load_store_contents()
    if store is not None:
        print(f"knowledge base: {len(store)} chunks")

    rng = np.random.default_rng(0)
    all_ok = True
    for m in args.models:
        print(f"== {m}")
        ok, j = check_model(m, samples, store, seed_check=not args.no_seed_check)
        all_ok &= ok
        if args.compare and j is not None and len(j):
            compare(m, j, rng)
            control(m)
    print("ALL CHECKS PASSED" if all_ok else "SOME CHECKS FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
