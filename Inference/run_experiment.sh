#!/usr/bin/env bash
# Runs the full FRAUD-ICL experiment grid, then the contamination check, in one shot.
#
# Runs standalone on any machine with a CUDA GPU; a scheduler job can call it as its body.
#
# Standalone usage (from anywhere; the script cd's to the repo root itself):
#   nohup bash Inference/run_experiment.sh > run.log 2>&1 &
#
# Resume after a crash/timeout: just run it again. Neither stage passes --overwrite, so
# both resume from the rows already written. See RESUME SEMANTICS below.

set -euo pipefail

# This script lives in Inference/; the repo root (main.py, Outputs/, Saved/) is its parent.
# Every path below is relative to the root, so resolve there rather than to $PWD.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"


# ---------------------------------------------------------------------------
# Tunables (override from the environment, e.g. `MODELS="qwen3" bash Inference/run_experiment.sh`)
# ---------------------------------------------------------------------------
MODELS="${MODELS:-neural_chat qwen3 mistral_instruct gemma}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./Outputs/full-run}"
SAMPLES_PATH="${SAMPLES_PATH:-./Outputs/total_samples.csv}"
DEVICE="${DEVICE:-cuda:0}"
QUANTIZATION="${QUANTIZATION:-4bit}"
CONTAM_N="${CONTAM_N:-300}"
# Separate tree from OUTPUT_ROOT -- the two stages must never write to the same directory.
CONTAM_ROOT="${CONTAM_ROOT:-./Outputs/contamination-check}"
# Keep model weights on the same (large) filesystem as the repo, never in $HOME.
SAVED_PATH="${SAVED_PATH:-./Saved}"
HF_CACHE_DIR="${HF_CACHE_DIR:-./Saved/hf_cache}"
# Grounding is scored after generation (stage 3), not inline: inline scoring adds ~2.3
# LettuceDetect passes per row between generations. Stage 3 re-scores from columns the run
# already saved, with identical results. Set BACKFILL_WORKERS=0 to skip it and score elsewhere.
# Rows per generation forward pass. Kept at 1: batching does not reproduce sequential
# output with 4-bit weights plus a repetition_penalty (3/8 prompts differed for gemma,
# 5/8 for mistral_instruct when we measured it). Raise only for runs that need not match
# sequential generation.
BATCH_SIZE="${BATCH_SIZE:-1}"
BACKFILL_WORKERS="${BACKFILL_WORKERS:-4}"
BACKFILL_MODES="${BACKFILL_MODES:-demos demos_query demos_query_def}"

# Xet would cache chunks outside the run's HF_HOME; plain HTTP downloads keep everything
# under --hf-cache-dir. main.py sets this too; exported here for any manual hf calls.
export HF_HUB_DISABLE_XET=1

echo "=== FRAUD-ICL full run ==="
echo "started:      $(date -Is)"
echo "host:         $(hostname)"
echo "repo:         $REPO_ROOT"
echo "models:       $MODELS"
echo "output root:  $OUTPUT_ROOT"
echo "device:       $DEVICE ($QUANTIZATION)"
echo

# ---------------------------------------------------------------------------
# Preflight: check inputs and the 4-bit kernel before the long run.
# bitsandbytes ships its own compiled kernels, so a passing torch matmul does NOT certify
# 4-bit quantization -- test Linear4bit explicitly.
# ---------------------------------------------------------------------------
echo "--- preflight: input data ---"
if [ ! -f "$SAMPLES_PATH" ]; then
    echo "ERROR: samples file not found: $SAMPLES_PATH" >&2
    exit 1
fi
echo "samples: $SAMPLES_PATH ($(wc -l < "$SAMPLES_PATH") lines)"
echo

echo "--- preflight: GPU + 4-bit kernel ---"
uv run python - <<'EOF'
import torch
assert torch.cuda.is_available(), "CUDA not available -- is this a GPU node? (check --gres / module load cuda)"
name = torch.cuda.get_device_name(0)
cap = torch.cuda.get_device_capability(0)
arch = f"sm_{cap[0]}{cap[1]}"
archs = torch.cuda.get_arch_list()
print(f"torch {torch.__version__} | cuda {torch.version.cuda}")
print(f"GPU: {name} ({arch}) | torch supports: {archs}")
assert arch in archs, f"torch build lacks {arch} kernels -- needs the cu128 index build"
from bitsandbytes.nn import Linear4bit
Linear4bit(64, 64, bias=False, compute_dtype=torch.float16, quant_type="nf4").cuda()(
    torch.randn(4, 64, device="cuda", dtype=torch.float16))
torch.cuda.synchronize()
print("4-bit quantization OK -- preflight passed")
EOF
echo

# ---------------------------------------------------------------------------
# RESUME SEMANTICS -- why neither stage passes --overwrite.
#
# --overwrite DELETES each existing def_*_outputs.csv before writing (os.remove in
# ExperimentManager.create_csv). That is correct for a clean first run, but on a cluster a
# job is far more likely to die from walltime/preemption than to complete first try, and a
# resubmit carrying --overwrite would discard every generation already produced.
#
# Omitting it makes both stages resumable: already-logged rows are counted and skipped, and
# fully-logged pipelines are skipped entirely. One exception: a kill in the middle of a row can
# leave a truncated CSV that the resume cannot read; that file then starts from the beginning,
# so delete the truncated file before re-running. To start again from scratch, delete the
# output tree instead:
#   rm -rf ./Outputs/full-run ./Outputs/contamination-check
#
# With --per-row-seed (set below) a resumed random_order/demo_order row draws the same
# demonstrations and permutations as an uninterrupted run.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Stage 1 -- full experiment grid
# ---------------------------------------------------------------------------
echo "--- stage 1/3: full experiment grid ---"
echo "start: $(date -Is)"
uv run python main.py \
  --samples-path "$SAMPLES_PATH" \
  --output-root "$OUTPUT_ROOT" \
  --models $MODELS \
  --retrievals cosine \
  --orders in_order random_order reverse_order demo_order \
  --batch-size "$BATCH_SIZE" \
  --definitions 0 1 2 3 \
  --formats new old \
  --no-hallucination-checker \
  --zero-shot \
  --per-row-seed \
  --run-knn \
  --knn-k 5 \
  --knn-output-dir "$OUTPUT_ROOT/kNN-Generation" \
  --saved-path "$SAVED_PATH" \
  --hf-cache-dir "$HF_CACHE_DIR" \
  --no-save-local-artifacts \
  --device "$DEVICE" \
  --quantization "$QUANTIZATION"
echo "stage 1 done: $(date -Is)"
echo

# ---------------------------------------------------------------------------
# Stage 2 -- contamination check.
# A standalone mode: main.py returns early for it, so it CANNOT be folded into stage 1's
# invocation. It ignores the definition/format/order/retrieval flags (its condition is fixed
# by construction) and writes to its own tree, so it cannot collide with stage 1's outputs.
# This stage regenerates BOTH arms for all four backbones. In the paper, the original arm is
# the main-grid rows of the 300 texts.
# ---------------------------------------------------------------------------
echo "--- stage 2/3: contamination check ---"
echo "start: $(date -Is)"
uv run python main.py --contamination-check \
  --models $MODELS \
  --samples-path "$SAMPLES_PATH" \
  --contamination-root "$CONTAM_ROOT" \
  --contamination-n "$CONTAM_N" \
  --saved-path "$SAVED_PATH" \
  --hf-cache-dir "$HF_CACHE_DIR" \
  --no-save-local-artifacts \
  --device "$DEVICE" \
  --quantization "$QUANTIZATION"
echo "stage 2 done: $(date -Is)"
echo

# ---------------------------------------------------------------------------
# Stage 3 -- grounding backfill.
# Stage 1 ran with --no-hallucination-checker, so the run logs carry empty grounding
# cells. This fills every requested mode from the persisted rows. Runs last because it
# needs no LLM: you can skip it (BACKFILL_WORKERS=0) and run Scripts/backfill_grounding.py
# on another machine.
# ---------------------------------------------------------------------------
if [ "$BACKFILL_WORKERS" -gt 0 ]; then
    echo "--- stage 3/3: grounding backfill ---"
    echo "start: $(date -Is)"
    uv run python Scripts/backfill_grounding.py \
      --run-dir "$OUTPUT_ROOT" \
      --modes $BACKFILL_MODES \
      --device "$DEVICE" \
      --workers "$BACKFILL_WORKERS"
    echo "stage 3 done: $(date -Is)"
else
    echo "--- stage 3/3: grounding backfill SKIPPED (BACKFILL_WORKERS=0) ---"
    echo "Run it wherever you like -- it needs no LLM:"
    echo "  python Scripts/backfill_grounding.py --run-dir $OUTPUT_ROOT --workers 4"
fi
echo

echo "=== ALL STAGES COMPLETE ==="
echo "finished: $(date -Is)"
echo
echo "Outputs to retrieve:"
du -sh "$OUTPUT_ROOT" 2>/dev/null || true
du -sh "$CONTAM_ROOT" 2>/dev/null || true
echo
echo "Parse the outputs with fraudicl.recovery before scoring; see README.md."
