# FRAUD-ICL

FRAUD-ICL is an open-source retrieval-augmented in-context-learning (RAG-ICL) pipeline for
multi-domain deception detection. For each text it retrieves labelled demonstrations from a
knowledge base built on the DiFrauD training split, orders them, builds a definition-augmented
prompt, runs an open-weight LLM that answers with a label and a rationale, and scores how far the
rationale departs from the demonstrations (LettuceDetect). Retrieval, ordering, deception
definition, prompt format and model are independent switches.

This repository holds the code and the exact evaluation samples of the paper *Evaluating
Explainable Multi-Domain Deception Detection Across Pipeline Choices* (Findings of EACL 2027;
arXiv link to follow).

**What the paper finds** (four 7-9B open-weight models, seven DiFrauD domains):

1. Influence-maximizing ordering (DEmO) gives no consistent benefit across models and metrics,
   and costs about 1.6-5x more compute than the simpler orderings.
2. Among the factors the experiment varies, domain source has by far the largest association
   with binary F1: one-way eta squared is 0.826 for domain, against 0.027 for the LLM backbone
   and 0.001 each for definition and ordering.
3. How far a rationale departs from the demonstrations does not reliably predict whether the
   prediction is correct: after controlling for domain and rationale length, the association
   survives for one of the four models (Gemma).

## Contents

| Path | What it is |
|---|---|
| `main.py`, `fraudicl/` | The pipeline: data, knowledge base, retrieval, ordering, prompting, generation, parsing, grounding |
| `Inference/run_experiment.sh` | The paper's run in one command: full grid, contamination check, grounding scores |
| `Scripts/run_random_retrieval.py`, `verify_random_retrieval.py` | The random-retrieval ablation and its check |
| `Scripts/backfill_grounding.py` | Grounding scores computed after generation (used by `run_experiment.sh`) |
| `Scripts/verify_batched_generation.py` | Check to pass before using `--batch-size` above 1 (the paper used 1) |
| `Analysis/baselines.py` | The supervised baselines: TF-IDF + LR, the GIST-embedding LR probe, fine-tuned DistilBERT |
| `Analysis/domain_variance.py` | The per-domain binary-F1 grid and one-way eta squared per factor |
| `Analysis/baseline_abstention.py` | The headline prompting comparison with abstentions counted as errors |
| `Analysis/common.py` | Compiles the generation CSVs (with parse-error recovery) and shared statistics |
| `Outputs/total_samples.csv` | The 3,000 evaluation texts; the `total_samples` column (500 / 1000 / 1500) gives sub-sample membership |
| `Outputs/random-retrieval/samples.csv` | The 500 queries of the random-retrieval ablation |
| `Outputs/contamination-check/samples_{original,perturbed}.csv` | The 300 contamination-check texts and their perturbed copies |

## Install

Requirements: Linux or Windows, [uv](https://docs.astral.sh/uv/), a CUDA GPU (all four models run
in 4-bit NF4 within 15 GB of VRAM), and about 15 GB of disk per model for weights.

```bash
git clone https://github.com/ReDASers/fraudicl-release.git
cd fraudicl-release
uv sync                         # Python 3.12 and the CUDA 12.8 build of PyTorch, from uv.lock
cp .env.example .env            # then put your Hugging Face token in .env
```

The token is needed only for `google/gemma-2-9b-it`, which is gated. Run everything from the
repository root with `uv run python ...`.

**Data.** DiFrauD (`redasers/difraud` on the Hugging Face Hub) is downloaded on first use into
`Saved/datasets/`. On its first run `main.py` builds the knowledge base from the DiFrauD *train*
split (texts of 5 to 900 words, 200-word chunks with 10 words of overlap,
`avsolatorio/GIST-small-Embedding-v0` embeddings, 123,597 chunks) and saves it to
`Saved/document_store`. The evaluation texts are the shipped samples above, drawn from the
validation split.

## Replicate the experiments

**Main grid, contamination check and grounding scores.** One command, resumable (re-run it and
finished rows are skipped; delete a CSV truncated by a kill before resuming):

```bash
bash Inference/run_experiment.sh
```

It runs three stages:

1. **Generate** (`Outputs/full-run/`): four models (`neural_chat qwen3 mistral_instruct gemma`),
   cosine retrieval with k = 4, orderings `in_order random_order reverse_order demo_order`
   (DEmO with all 24 permutations and 4 candidates kept), Definitions 0-3, both prompt formats
   (`new` = FRAUD-ICL, `old` = RADDICL), zero-shot and the kNN baseline. Equivalent to:

   ```bash
   uv run python main.py --samples-path Outputs/total_samples.csv --output-root Outputs/full-run \
     --models neural_chat qwen3 mistral_instruct gemma --retrievals cosine \
     --orders in_order random_order reverse_order demo_order --definitions 0 1 2 3 --formats new old \
     --zero-shot --per-row-seed --no-hallucination-checker --batch-size 1 \
     --run-knn --knn-k 5 --knn-output-dir Outputs/full-run/kNN-Generation \
     --saved-path Saved --hf-cache-dir Saved/hf_cache --no-save-local-artifacts \
     --device cuda:0 --quantization 4bit
   ```

   One CSV per model and (definition, format, retrieval) lands in
   `Outputs/full-run/<Model>-Generation/def_{definition}_{format}_{retrieval}_outputs.csv`.
2. **Contamination check** (`Outputs/contamination-check/`): Definition 1, FRAUD-ICL format,
   cosine, in-order, and zero-shot, on the 300 shipped texts and their perturbed copies.
3. **Grounding scores**: LettuceDetect on the saved rows, for three reference contexts
   (`demos`, `demos_query`, `demos_query_def`); the paper reports `demos`.

**Random retrieval** (Definition 1, in-order, k = 4, the 500 shipped queries). Needs the main
grid's cosine files for the comparison:

```bash
uv run python Scripts/run_random_retrieval.py      # --prepare-only checks inputs without generating
uv run python Scripts/verify_random_retrieval.py --compare
```

**Scoring.** Parse each output with `fraudicl.recovery`, the parse-error recovery the paper
applies everywhere, so that "valid response" means the same thing in every analysis.
`Analysis/common.py` does this for you.

**Determinism.** `--per-row-seed` seeds every row from (seed 99, row index, pipeline), so random
and DEmO orderings and random retrieval draw the same demonstrations after an interruption.
Decoding is greedy with batch size 1. Token-identical output is only expected on the same GPU
model and library versions; the paper's rows were generated on NVIDIA RTX 5090, RTX 5080, A10 and
A100 GPUs (paper, App. E.2), so another machine may flip near-tied tokens.

## Analysis

The scripts in `Analysis/` run on CPU, except DistilBERT fine-tuning, which needs a GPU.
Results go to `Outputs/analysis/<script>/`.

```bash
uv run python Analysis/baselines.py                     # TF-IDF + LR and GIST + LR
uv run python Analysis/baselines.py --only distilbert   # fine-tuned DistilBERT (GPU, about 20-60 min)
uv run python Analysis/domain_variance.py               # F1 grid, eta squared, domain means
uv run python Analysis/baseline_abstention.py           # abstention-counted headline comparison
```

The baselines need only DiFrauD and `Outputs/total_samples.csv`. They train on the DiFrauD
train split, so their numbers do not depend on the LLM runs. The other two scripts read the
main grid in `Outputs/full-run/`. On first use they compile each model's CSVs into
`Outputs/compiled/<Model>_data.parquet`, applying parse-error recovery, and recompile when a
generation CSV changes. On the paper's generations, these scripts reproduce its reported
numbers exactly.

**Scope.** The repository holds the code for generation, grounding, the random-retrieval
ablation and the three analyses above. The scripts behind the paper's other figures and
appendix statistics (ordering and definition tests, departure analyses, coverage tables, the
contamination statistics) are not included.

## Attribution

```bibtex
@inproceedings{denq-verma-2027-fraud-icl,
  title     = {Evaluating Explainable Multi-Domain Deception Detection Across Pipeline Choices},
  author    = {Denq, Christopher and Verma, Rakesh M},
  booktitle = {Findings of the Association for Computational Linguistics: EACL 2027},
  year      = {2027},
  url       = {https://github.com/ReDASers/fraudicl-release}
}
```

FRAUD-ICL builds on RADDICL (Boumber et al., 2024) and evaluates on DiFrauD (Boumber et al.,
2024); please cite them when you use the RADDICL prompt format or the dataset. DEmO ordering
follows Guo et al. (2024); grounding uses LettuceDetect (Kovács and Recski, 2025).

**Data terms.** We do not redistribute DiFrauD in full. The shipped samples are provided for
exact reproduction and remain subject to the source datasets' licenses (DiFrauD is distributed
under the MIT License; see its dataset card). They may contain personal or offensive content.

**License.** MIT, see `LICENSE`.
