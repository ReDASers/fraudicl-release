"""Backfill LettuceDetect grounding columns into run-log CSVs after the fact.

Why this exists
---------------
Scoring every grounding mode inline costs ~2.3 LettuceDetect forward passes per
row (``check_all`` dedups identical context assemblies), all on the critical path
between one generation and the next. ``Inference/run_experiment.sh`` therefore
generates with ``--no-hallucination-checker`` and runs this script afterwards. It
re-scores from data the run already saved: ``raw_answer``, ``retrieved_examples``,
``prompt``, ``definition``.

No LLM generation is re-run. The arms produced here are identical to what the
inline path would have written -- ``check_all`` is reused rather than
reimplemented, so context assembly and the scored-text transform cannot drift.

Scored text
-----------
Grounding is scored over ``scored_answer(raw_answer)`` -- the full answer with
thinking tags stripped -- NOT the parsed ``reasoning`` field. The departure
fraction is a share of answer characters, so the denominator must be the text the
detector actually saw; ``reasoning`` omits the verdict sentence.

Usage
-----
    # verify against rows that already carry the primary mode, then stop
    python Scripts/backfill_grounding.py --run-dir Outputs/full-run --check-only

    # backfill every CSV under a run directory
    python Scripts/backfill_grounding.py --run-dir Outputs/full-run

    # one model, explicit device, smoke test
    python Scripts/backfill_grounding.py \
        --run-dir Outputs/full-run/Mistral-Generation --device cuda:0 --limit 50
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

import fraudicl.global_config as global_config
from fraudicl.evaluationmanager import HallucinationChecker, scored_answer

# Definition text depends on the backbone's prompt profile, so the re-scored
# "demos+query+def" context matches what that model was actually shown.
DIR_TO_PROFILE = {
    "Gemma-Generation": "gemma",
    "Mistral-Generation": "mistral_instruct",
    "Neural-Generation": "neural_chat",
    "Qwen-Generation": "qwen3",
}


def prompt_class_for(path: Path):
    """Resolve the prompt class from the model's output directory name."""
    for part in reversed(path.parts):
        key = DIR_TO_PROFILE.get(part)
        if key:
            return global_config.PromptRegistry.get(key)
    return global_config.Prompt


def definition_text(prompt_class, definition_type: int) -> Optional[str]:
    """Definition injected into the prompt; -1/0 carry none (see PromptManager)."""
    if definition_type in (-1, 0):
        return None
    return getattr(prompt_class, "DEFINITION_{}".format(int(definition_type)), None)


def demo_contents(retrieved_examples: Any) -> List[str]:
    """Extract demonstration text from the persisted retrieved_examples JSON."""
    if not isinstance(retrieved_examples, str) or not retrieved_examples.strip():
        return []
    try:
        parsed = json.loads(retrieved_examples)
    except json.JSONDecodeError:
        return []
    return [d["content"] for d in parsed if isinstance(d, dict) and d.get("content")]


def score_row(checker, row, prompt_class, modes) -> Dict[str, Any]:
    """All requested modes for one row, via the same call the inline path uses."""
    return checker.check_all(
        reasoning=scored_answer(row["raw_answer"]),
        context_docs=demo_contents(row["retrieved_examples"]),
        question=str(row["prompt"]),
        definition=definition_text(prompt_class, int(row["definition"])),
        modes=modes,
    )


def backfill_csv(
    path: Path,
    checker: HallucinationChecker,
    modes: List[str],
    check_only: bool,
    limit: Optional[int],
    tolerance: float,
) -> Dict[str, Any]:
    """Re-score one run-log CSV. Returns a summary dict."""
    df = pd.read_csv(path)
    prompt_class = prompt_class_for(path)
    cfg = global_config.LettuceDetect

    # Not every *_outputs.csv is a generation log: the kNN baseline writes
    # knn_<retrieval>_k<k>_outputs.csv into the same tree and carries no rationale
    # to score. Skip anything without the columns a re-score needs rather than
    # failing the whole run on it.
    required = {"raw_answer", "retrieved_examples", "prompt", "definition"}
    missing = required - set(df.columns)
    if missing:
        print("    skipping {} (not a generation log; no {})".format(
            path.name, ", ".join(sorted(missing))))
        return {
            "file": str(path), "rows": len(df), "scored": 0,
            "compared": 0, "mismatches": 0, "written": False, "skipped": True,
        }

    # Zero-shot rows carry no retrieved context, so the inline path never scored
    # them; skip them here too rather than inventing values.
    demos = df["retrieved_examples"].map(demo_contents)
    scorable = [i for i in df.index if len(demos[i]) > 0]
    if limit is not None:
        scorable = scorable[:limit]

    primary_col = cfg.column_name("hallucinated_fraction", cfg.PRIMARY_CONTEXT_MODE)

    mismatches = 0
    compared = 0
    updates: Dict[str, Dict[int, Any]] = {}

    for n, idx in enumerate(scorable, start=1):
        row = df.loc[idx]
        scored = score_row(checker, row, prompt_class, modes)

        # Free correctness check: where the inline pass already wrote a primary
        # value, the recomputed one must match. Drift here means the scored text
        # or the context assembly is wrong -- stop rather than write.
        if primary_col in df.columns and pd.notna(row.get(primary_col)):
            compared += 1
            delta = abs(float(scored[primary_col]) - float(row[primary_col]))
            if delta > tolerance:
                mismatches += 1
                if mismatches <= 5:
                    print(
                        "    MISMATCH row {}: stored={!r} recomputed={!r} (delta {:.6f})".format(
                            idx, row[primary_col], scored[primary_col], delta
                        )
                    )

        if not check_only:
            for col, val in scored.items():
                updates.setdefault(col, {})[idx] = val

        if n % 500 == 0 or n == len(scorable):
            print("    {}: {}/{} rows".format(path.name, n, len(scorable)), flush=True)

    summary = {
        "file": str(path),
        "rows": len(df),
        "scored": len(scorable),
        "compared": compared,
        "mismatches": mismatches,
        "written": False,
        "skipped": False,
    }

    if check_only or not updates:
        return summary

    if mismatches:
        print("    REFUSING to write {}: {} mismatched rows".format(path.name, mismatches))
        return summary

    # Back the file up once, write to a sibling temp file, then replace, so an
    # interrupted run cannot leave a half-written log behind.
    backup = path.with_suffix(path.suffix + ".bak")
    if not backup.exists():
        shutil.copy2(path, backup)
    # Build each column as an object Series before assigning. Columns read from a
    # partially-written log are often inferred as float64 (all-empty), and writing
    # bools or JSON strings into such a column raises rather than upcasting.
    for col, mapping in updates.items():
        if col in df.columns:
            base = df[col].astype(object)
        else:
            base = pd.Series(pd.NA, index=df.index, dtype=object)
        for idx, val in mapping.items():
            base.at[idx] = val
        df[col] = base
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)
    summary["written"] = True
    return summary


def parse_args() -> argparse.Namespace:
    cfg = global_config.LettuceDetect
    parser = argparse.ArgumentParser(
        description="Backfill LettuceDetect grounding columns into run-log CSVs.",
    )
    parser.add_argument(
        "--run-dir", type=Path, required=True,
        help="Run directory, or a single model's *-Generation directory.",
    )
    parser.add_argument(
        "--modes", nargs="+", default=list(cfg.DEFAULT_SCORED_MODES),
        help="Context modes to write (slugs or raw mode strings).",
    )
    parser.add_argument(
        "--device", default=None,
        help='e.g. "cuda:0" or "cpu". Default: LettuceDetect auto-selects.',
    )
    parser.add_argument(
        "--check-only", action="store_true",
        help="Verify recomputed primary values against stored ones; write nothing.",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Score at most N rows per file (smoke test).",
    )
    parser.add_argument(
        "--tolerance", type=float, default=1e-6,
        help="Max allowed drift when comparing recomputed primary values.",
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help=(
            "Score this many CSVs concurrently, one process per file. LettuceDetect's "
            "predict_prompt_batch is a Python loop, not a real batched forward pass, so "
            "file-level parallelism is where the speedup comes from. Each worker loads "
            "its own detector (~600MB), so keep workers x 600MB inside VRAM on cuda."
        ),
    )
    return parser.parse_args()


def _worker(job):
    """Run one file in a child process. Returns the same summary dict."""
    path, modes, device, check_only, limit, tolerance = job
    checker = HallucinationChecker(device=device)
    return backfill_csv(path, checker, modes, check_only, limit, tolerance)


def main() -> None:
    args = parse_args()
    modes = global_config.LettuceDetect.resolve_modes(args.modes)
    # Generation logs are def_<n>_<tag>_<retrieval>_outputs.csv; the kNN baseline
    # writes knn_*_outputs.csv into the same tree and has no rationale to score.
    files = sorted(p for p in args.run_dir.rglob("def_*_outputs.csv"))
    if not files:
        raise SystemExit("No def_*_outputs.csv found under {}".format(args.run_dir))

    print("Modes: {}".format(", ".join(modes)))
    print("Files: {}".format(len(files)))

    if args.workers > 1:
        import concurrent.futures as cf

        jobs = [
            (p, modes, args.device, args.check_only, args.limit, args.tolerance)
            for p in files
        ]
        print("Workers: {}".format(args.workers))
        summaries = []
        with cf.ProcessPoolExecutor(max_workers=args.workers) as pool:
            for s in pool.map(_worker, jobs):
                summaries.append(s)
    else:
        checker = HallucinationChecker(device=args.device)
        summaries = []
        for path in files:
            print("  {}".format(path))
            summaries.append(
                backfill_csv(path, checker, modes, args.check_only, args.limit, args.tolerance)
            )

    print("\n=== summary ===")
    total_mismatches = 0
    for s in summaries:
        total_mismatches += s["mismatches"]
        if s.get("skipped"):
            state = "not a generation log"
        elif args.check_only:
            state = "check"
        else:
            state = "written" if s["written"] else "SKIPPED"
        print(
            "{}: {} scored, {} compared, {} mismatched [{}]".format(
                Path(s["file"]).name, s["scored"], s["compared"], s["mismatches"], state
            )
        )
    if total_mismatches:
        raise SystemExit(
            "{} mismatched rows -- not safe to trust this backfill".format(total_mismatches)
        )
    print("All recomputed primary values matched stored values.")


if __name__ == "__main__":
    main()
