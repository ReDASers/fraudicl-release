"""Parse-error recovery for raw generation outputs.

The inline parser (``fraudicl.evaluationmanager.AnswerParser``) marks a row invalid when it
cannot map an output to a score: ``pred_tag`` in {unknown, PARSER ERROR, both}, ``pred_label``
== -1, ``valid`` == False. This module re-reads ``raw_answer`` and recovers only genuine parse
misses.

There are two causes of an invalid row:

* ``unknown``: the model's own verdict was 'unknown', an abstention. This is a real answer, not
  a parse failure, so it is not relabeled. Scanning its reasoning for the word 'deceptive' would
  invent a verdict the model never gave.
* ``PARSER ERROR``: the parser failed to extract a label that is present (e.g. the model wrote
  "'not deceptive'" wrapped in quotes, which broke the label-first regex). These rows are
  recovered.

Policy:

* recover only ``pred_tag`` in {PARSER ERROR, both}; leave 'unknown' invalid;
* 'both' (deceptive AND non-deceptive) resolves to deceptive;
* invalid rows are kept either way (``valid`` flips to True only on recovery);
* residual unrecoverable parse errors normalize to 'unknown', so no raw 'PARSER ERROR' tag
  reaches later analysis as a separate category.

Prompts end in 'Label:' and these models answer label-first, so the first verdict token wins (an
explicit 'Label: <x>' anchor first); if that first verdict is itself 'unknown', we abstain.

Every analysis of the outputs (main grid, contamination check, random retrieval) uses this
module, so "valid response" means the same thing in each.
"""
from __future__ import annotations

import re
from typing import Any, Optional, Tuple

import pandas as pd

import fraudicl.global_config as global_config

LABEL_TO_SCORE = {v: k for k, v in global_config.Database.LABEL_MAPPING.items()}

_DEC = r"deceptive|delective|decessive"  # deceptive + the pipeline's known typos
_VERDICT = rf"unknown|non[-\s]?(?:{_DEC})|not\s+(?:{_DEC})|un-?(?:{_DEC})|(?:{_DEC})"
_ANCHOR_RE = re.compile(
    r"(?:label|prediction|answer)\s*(?:\*\*)?\s*[:\-]\s*[\"'*]*\s*(" + _VERDICT + ")")
_VERDICT_RE = re.compile(_VERDICT)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

RECOVERABLE_TAGS = {"parser error", "both"}


def scan_verdict(text: str) -> Optional[str]:
    """First verdict token in a cleaned answer -> 'deceptive'|'non-deceptive'|'unknown'|None."""
    m = _ANCHOR_RE.search(text)          # explicit 'Label: <x>' wins
    tok = m.group(1) if m else None
    if tok is None:
        m = _VERDICT_RE.search(text)     # else first verdict mention (label-first prompting)
        tok = m.group(0) if m else None
    if tok is None:
        return None
    tok = tok.strip()
    if tok.startswith("unknown"):
        return "unknown"
    return "non-deceptive" if tok.startswith(("non", "not", "un")) else "deceptive"


def recover_label(raw_answer: Any, orig_pred_tag: Any) -> Optional[str]:
    """Recovered verdict for one invalid row, or None to leave it invalid."""
    tag = orig_pred_tag.strip().lower() if isinstance(orig_pred_tag, str) else ""
    if tag not in RECOVERABLE_TAGS:
        return None                      # e.g. 'unknown' -> genuine abstention
    if tag == "both":
        return "deceptive"               # 'both' -> deceptive (policy above)
    if not isinstance(raw_answer, str) or not raw_answer.strip():
        return None
    cleaned = _THINK_RE.sub(" ", raw_answer.lower()).replace("<end_of_turn>", " ")
    verdict = scan_verdict(cleaned)
    return verdict if verdict in ("deceptive", "non-deceptive") else None


def recover_frame(df: pd.DataFrame) -> Tuple[pd.DataFrame, int, int]:
    """Apply parse-error recovery to a raw-outputs frame, in place.

    Returns ``(df, n_recovered, n_normalized)``. ``n_normalized`` counts residual
    unrecoverable parse errors relabeled to 'unknown' (they stay invalid).
    """
    valid_mask = df["valid"].astype(str).str.lower().eq("true")
    recovered = 0
    for i in df.index[~valid_mask]:
        label = recover_label(df.at[i, "raw_answer"], df.at[i, "pred_tag"])
        if label is None:
            continue
        score = LABEL_TO_SCORE[label]
        df.at[i, "pred_tag"] = label
        df.at[i, "pred_label"] = score
        df.at[i, "valid"] = True
        if "correct" in df.columns:
            df.at[i, "correct"] = bool(score == int(df.at[i, "true_label"]))
        recovered += 1

    residual = df.index[
        (~df["valid"].astype(str).str.lower().eq("true"))
        & df["pred_tag"].astype(str).str.strip().str.lower().isin(list(RECOVERABLE_TAGS))
    ]
    df.loc[residual, "pred_tag"] = "unknown"
    return df, recovered, int(len(residual))


def load_recovered(path) -> Tuple[pd.DataFrame, int]:
    """Read a raw outputs CSV and apply parse-error recovery. Returns ``(df, n_recovered)``."""
    df, recovered, _ = recover_frame(pd.read_csv(path))
    return df, recovered
