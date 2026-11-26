from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Sequence

os.environ["HF_HUB_DISABLE_XET"] = "1"

import pandas as pd
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from fraudicl import global_config
from fraudicl.datamanager import DataConfig, DataManager, DataSampler
from fraudicl.datastorage import VectorDBConfig, VectorDBManager
from fraudicl.embedder import EmbeddingConfig, EmbeddingManager
from fraudicl.evaluationmanager import EvaluationManager, HallucinationChecker
from fraudicl.experimentmanager import ExperimentManager
from fraudicl.helper import memclear
from fraudicl.llmmanager import LLMConfig, LLMManager
from fraudicl.ordermanager import OrderManager
from fraudicl.pipelinemanager import PipelineConfig, PipelineManager
from fraudicl.promptmanager import PromptManager
from fraudicl.retrievalmanager import RetrievalManager


DEFAULT_MODELS = ["neural_chat", "qwen3", "mistral_instruct", "gemma"]
DEFAULT_RETRIEVALS = ["cosine"]
DEFAULT_ORDERS = ["in_order", "random_order", "reverse_order", "demo_order"]
DEFAULT_DEFINITIONS = [0, 1, 2, 3]
DEFAULT_FORMATS = ["new", "old"]

KNN_HEADERS = [
    "retrieval",
    "k",
    "domain",
    "true_label",
    "pred_label",
    "valid",
    "correct",
    "retrieved_labels",
    "n_retrieved",
]


@dataclass(frozen=True)
class RuntimeConfig:
    dataset_path: str
    hf_revision: str
    split: str
    samples_path: Path
    document_store: str
    output_root: Path
    processing_output_dir: Path | None
    saved_path: Path | None
    hf_cache_dir: Path | None
    save_local_artifacts: bool
    cleanup_download_cache: bool
    device: str
    quantization: str | None
    max_new_tokens: int | None
    purge_model_cache: bool
    models: list[str]
    retrievals: list[str]
    orders: list[str]
    definitions: list[int]
    formats: list[str]
    overwrite: bool
    per_row_seed: bool
    batch_size: int
    hallucination_checker: bool
    grounding_modes: list[str]
    zero_shot: bool
    run_knn: bool
    knn_k: int
    knn_output_dir: Path


class KNNLabeler:
    def __init__(self, retriever: RetrievalManager, embeddings: dict[str, list[float]], k: int) -> None:
        self.retriever = retriever
        self.embeddings = embeddings
        self.k = k

    def predict(self, query_text: str, retrieval_strategy: str) -> dict:
        embedding = self.embeddings.get(query_text)
        if embedding is None:
            return {"pred_label": -1, "valid": False, "retrieved_labels": [], "n_retrieved": 0}

        result = self.retriever.run(
            user_query=embedding,
            retrieval_strategy=retrieval_strategy,
            top_k=self.k,
        )
        docs = result.get("retrieved_documents", [])
        if not docs:
            return {"pred_label": -1, "valid": False, "retrieved_labels": [], "n_retrieved": 0}

        retrieved_labels = [int(doc.meta.get("label", -1)) for doc in docs]
        valid_labels = [label for label in retrieved_labels if label in (0, 1)]
        if not valid_labels:
            return {
                "pred_label": -1,
                "valid": False,
                "retrieved_labels": retrieved_labels,
                "n_retrieved": len(docs),
            }

        counts = Counter(valid_labels)
        pred_label = 0 if counts.get(0, 0) > counts.get(1, 0) else 1
        return {
            "pred_label": pred_label,
            "valid": True,
            "retrieved_labels": retrieved_labels,
            "n_retrieved": len(docs),
        }

    def run_experiment(
        self,
        queries: Sequence,
        retrieval_strategies: Sequence[str],
        output_dir: Path,
        overwrite: bool,
    ) -> pd.DataFrame:
        output_dir.mkdir(parents=True, exist_ok=True)
        all_frames = []

        for retrieval in retrieval_strategies:
            filepath = output_dir / f"knn_{retrieval}_k{self.k}_outputs.csv"
            append_mode = filepath.exists() and not overwrite
            completed_rows = count_logged_rows(filepath) if append_mode else 0
            print(f"\n[{retrieval.upper()} | k={self.k}] {'Appending' if append_mode else 'Writing'}")
            if completed_rows > 0:
                print(f"Resuming after {completed_rows} previously logged queries.")
            pending_queries = list(queries)[completed_rows:]

            rows = []
            for query_doc in tqdm(pending_queries, desc=retrieval):
                res = self.predict(query_doc.content, retrieval)
                true_label = int(query_doc.meta.get("label", -1))
                pred_label = res["pred_label"]
                valid = res["valid"]
                correct = pred_label == true_label if valid else False
                rows.append(
                    [
                        retrieval,
                        self.k,
                        query_doc.meta.get("domain", "unknown"),
                        true_label,
                        pred_label,
                        valid,
                        correct,
                        str(res["retrieved_labels"]),
                        res["n_retrieved"],
                    ]
                )

            with filepath.open("a" if append_mode else "w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                if not append_mode:
                    writer.writerow(KNN_HEADERS)
                writer.writerows(rows)

            df = pd.read_csv(filepath)
            valid_df = df[df["valid"].astype(bool)]
            acc = accuracy_score(valid_df["true_label"], valid_df["pred_label"]) if not valid_df.empty else 0.0
            print(f"  -> {len(df)} rows ({len(valid_df)} valid) | Accuracy: {acc:.4f} -> {filepath}")
            all_frames.append(df)

        return pd.concat(all_frames, ignore_index=True) if all_frames else pd.DataFrame()


def count_logged_rows(filepath: Path) -> int:
    """Count data rows already written so interrupted runs resume instead of duplicating."""
    if not filepath.exists():
        return 0
    try:
        return len(pd.read_csv(filepath, usecols=[KNN_HEADERS[0]]))
    except (pd.errors.EmptyDataError, ValueError):
        return 0


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset-path", default=global_config.General.LOCAL_DIFRAUD_PATH)
    parser.add_argument("--hf-revision", default="main")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--document-store", default="document_store")
    parser.add_argument("--processing-output-dir", type=Path)
    parser.add_argument("--saved-path", type=Path)
    parser.add_argument("--hf-cache-dir", type=Path)
    parser.add_argument(
        "--save-local-artifacts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Save a local copy of each quantized model under --saved-path so later runs load it from disk. "
            "--no-save-local-artifacts skips only that copy: the DiFrauD files and the document store are "
            "always saved under --saved-path and reused by every later run."
        ),
    )
    parser.add_argument("--cleanup-download-cache", action="store_true")
    parser.add_argument("--device", default=global_config.General.DEVICE_TYPE)
    parser.add_argument("--quantization", default="4bit", choices=["4bit", "8bit", "none"])
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Override the per-model max_new_tokens generation cap (profile default: 2048).",
    )
    parser.add_argument(
        "--purge-model-cache",
        action="store_true",
        help=(
            "After each model's experiments finish, delete its HF download cache and "
            "local quantized snapshot to reclaim disk. Skip if you plan to rerun that "
            "model on this machine."
        ),
    )
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--retrievals", nargs="+", default=DEFAULT_RETRIEVALS)
    parser.add_argument("--orders", nargs="+", default=DEFAULT_ORDERS)
    parser.add_argument("--definitions", nargs="+", type=int, default=DEFAULT_DEFINITIONS)
    parser.add_argument("--formats", nargs="+", choices=["new", "old"], default=DEFAULT_FORMATS)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--per-row-seed",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Reseed the RNG per row from (seed, row, pipeline) so random_order/demo_order are "
            "bitwise-reproducible even across resume. Off by default; it changes the random "
            "draws relative to a run without it. Inference/run_experiment.sh turns it on."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help=(
            "Rows generated per forward pass (default 1, as in the paper). With 4-bit "
            "weights and a repetition penalty, batched output does not match sequential "
            "output; check with Scripts/verify_batched_generation.py before raising it."
        ),
    )
    parser.add_argument("--hallucination-checker", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--grounding-modes",
        nargs="+",
        default=list(global_config.LettuceDetect.DEFAULT_SCORED_MODES),
        choices=sorted(global_config.LettuceDetect.MODE_SLUGS.values())
                + list(global_config.LettuceDetect.CONTEXT_MODES),
        metavar="MODE",
        help=(
            "LettuceDetect reference contexts to score each response against. "
            "Each mode adds a block of grounding columns to the output CSV. "
            "demos: retrieved demonstrations only -- reproduces the published "
            "demonstration-relative departure, and owns the unsuffixed columns "
            "(grounded, hallucinated_fraction, ...). "
            "demos_query: also counts the evaluated text as evidence. "
            "demos_query_def: also counts the deception definition, matching what "
            "the system prompt tells the model to reason from; collapses onto "
            "demos_query for definition types -1/0, which carry no definition. "
            "Extra modes are suffixed (e.g. hallucinated_fraction_demos_query). "
            "Default: all three."
        ),
    )
    parser.add_argument("--zero-shot", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--run-knn", action="store_true")
    parser.add_argument("--knn-k", type=int, default=5)


def build_config_from_args(args: argparse.Namespace) -> RuntimeConfig:
    return RuntimeConfig(
        dataset_path=args.dataset_path,
        hf_revision=args.hf_revision,
        split=args.split,
        samples_path=args.samples_path,
        document_store=args.document_store,
        output_root=args.output_root,
        processing_output_dir=args.processing_output_dir,
        saved_path=args.saved_path,
        hf_cache_dir=args.hf_cache_dir,
        save_local_artifacts=args.save_local_artifacts,
        cleanup_download_cache=args.cleanup_download_cache,
        device=args.device,
        quantization=None if args.quantization == "none" else args.quantization,
        max_new_tokens=args.max_new_tokens,
        purge_model_cache=args.purge_model_cache,
        models=args.models,
        retrievals=args.retrievals,
        orders=args.orders,
        definitions=args.definitions,
        formats=args.formats,
        overwrite=args.overwrite,
        per_row_seed=args.per_row_seed,
        batch_size=args.batch_size,
        hallucination_checker=args.hallucination_checker,
        grounding_modes=args.grounding_modes,
        zero_shot=args.zero_shot,
        run_knn=args.run_knn,
        knn_k=args.knn_k,
        knn_output_dir=args.knn_output_dir,
    )


def parse_args() -> tuple[RuntimeConfig, argparse.Namespace]:
    parser = argparse.ArgumentParser(description="Run FRAUD-ICL workflow experiments.")
    add_common_arguments(parser)
    parser.add_argument("--samples-path", type=Path, default=Path("./Outputs/total_samples.csv"))
    parser.add_argument("--output-root", type=Path, default=Path("./Outputs"))
    parser.add_argument("--knn-output-dir", type=Path, default=Path("./Outputs/kNN-Generation"))
    parser.add_argument(
        "--contamination-check",
        action="store_true",
        help=(
            "Run the data-contamination probe INSTEAD of the standard experiment grid. "
            "Draws a class-stratified subset of --samples-path, writes a label-preserving "
            "perturbed copy, and runs the canonical FRAUD-ICL condition (Def 1, new, cosine, "
            "in-order) plus zero-shot on both arms. Definition/format/order/retrieval flags are "
            "ignored -- the probe's condition is fixed by construction. Results are written "
            "under --contamination-root and never touch the main run's output tree."
        ),
    )
    parser.add_argument(
        "--contamination-root",
        type=Path,
        default=Path("./Outputs/contamination-check"),
        help="Output tree for --contamination-check (default: ./Outputs/contamination-check).",
    )
    parser.add_argument(
        "--contamination-n",
        type=int,
        default=300,
        help="Size of the class-stratified contamination subset (default: 300).",
    )
    parser.add_argument(
        "--contamination-seed",
        type=int,
        default=99,
        help="Seed for the contamination subset draw and its perturbations (default: 99).",
    )
    args = parser.parse_args()
    return build_config_from_args(args), args


def apply_global_paths(config: RuntimeConfig) -> None:
    global_config.General.DEVICE_TYPE = config.device
    global_config.General.LOCAL_PROCESSING_PATH = str(processing_output_path(config))
    if config.saved_path is not None:
        global_config.General.LOCAL_SAVED_PATH = str(config.saved_path)
    saved_path(config).mkdir(parents=True, exist_ok=True)
    cache_dir = hf_cache_path(config)
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache_dir)
    os.environ["HF_HUB_CACHE"] = str(cache_dir / "hub")
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(cache_dir / "hub") # legacy just in case
    os.environ["TRANSFORMERS_CACHE"] = str(cache_dir / "transformers")
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    os.environ["HF_XET_CACHE"] = str(cache_dir / "xet")


def processing_output_path(config: RuntimeConfig) -> Path:
    return config.processing_output_dir or config.output_root / "Processing"


def saved_path(config: RuntimeConfig) -> Path:
    return config.saved_path or Path(global_config.General.LOCAL_SAVED_PATH)


def hf_cache_path(config: RuntimeConfig) -> Path:
    return config.hf_cache_dir or saved_path(config) / "hf_cache"


def local_dataset_path(config: RuntimeConfig) -> Path:
    dataset_tag = config.dataset_path.replace("/", "_")
    return saved_path(config) / "datasets" / dataset_tag / config.hf_revision


def load_queries(config: RuntimeConfig) -> list:
    load_split(config, config.split)
    return DataSampler.load_from_csv(str(config.samples_path))


def load_split(config: RuntimeConfig, split: str) -> DataManager:
    data_config = DataConfig(
        dataset_path=config.dataset_path,
        hf_revision=config.hf_revision,
        local_dataset_path=str(local_dataset_path(config)),
        save_downloads=True,  # the dataset is always kept locally; --save-local-artifacts covers model snapshots only
        processing_output_path=str(processing_output_path(config)),
    )
    data_manager = DataManager(config=data_config)
    data_manager.process_all_domains(split)
    return data_manager


def setup_embedding_and_store(config: RuntimeConfig) -> tuple[EmbeddingManager, list]:
    embedder = EmbeddingManager(config=EmbeddingConfig(device_type=config.device))

    documents = load_or_build_document_store(config, embedder)
    return embedder, documents


def load_or_build_document_store(config: RuntimeConfig, embedder: EmbeddingManager) -> list:
    storage_manager = VectorDBManager(
        embedding_model=embedder,
        config=VectorDBConfig(device_type=config.device),
    )
    local_store = saved_path(config) / config.document_store
    if local_store.exists():
        return storage_manager.load_document_store(config.document_store)

    print(f"Local document store not found at {local_store}.")
    print("Building document store locally from DiFrauD train split.")

    train_manager = load_split(config, "train")
    training_documents = getattr(train_manager, "train")
    storage_manager.setup_vector_database(training_documents)
    documents = storage_manager.document_store.filter_documents()
    # Always saved: the store takes minutes to build and later runs (and the random-retrieval and
    # contamination runners) load it from here. --save-local-artifacts covers model snapshots only.
    storage_manager.save_document_store(config.document_store)
    return documents


def model_output_path(output_root: Path, model_key: str) -> Path:
    labels = {
        "neural_chat": "Neural-Generation",
        "qwen3": "Qwen-Generation",
        "qwen": "Qwen-Generation",
        "mistral_instruct": "Mistral-Generation",
        "mistral": "Mistral-Generation",
        "gemma": "Gemma-Generation",
        "gemma2_it": "Gemma-Generation",
    }
    return output_root / labels.get(model_key, f"{model_key}-Generation")


def _logged_rows_for_order(filepath: Path, order: str) -> int:
    """Rows already logged for a given order within an output CSV (mirrors
    ExperimentManager._count_logged_rows so completeness is judged the same way resume is)."""
    if not filepath.exists():
        return 0
    try:
        logged = pd.read_csv(filepath, usecols=["order"])
    except (pd.errors.EmptyDataError, ValueError):
        return 0
    return int((logged["order"] == order).sum())


def model_outputs_complete(config: RuntimeConfig, model_key: str, num_inputs: int, system_formats: list[bool]) -> bool:
    """True only if every output CSV this model would produce is already fully logged, so the
    model can be skipped WITHOUT downloading it. Replicates the multi_passthrough grid (including
    the def-0 + old skip) and the zero-shot pass, requiring num_inputs rows for every order."""
    out_dir = model_output_path(config.output_root, model_key)
    for definition_type in config.definitions:
        for use_legacy_format in system_formats:
            if definition_type == 0 and use_legacy_format:
                continue
            tag = "old" if use_legacy_format else "new"
            for retrieval in config.retrievals:
                filepath = out_dir / f"def_{definition_type}_{tag}_{retrieval}_outputs.csv"
                for order in config.orders:
                    if _logged_rows_for_order(filepath, order) < num_inputs:
                        return False
    if config.zero_shot:
        zero_shot_path = out_dir / "def_-1_new_none_outputs.csv"
        if _logged_rows_for_order(zero_shot_path, "none") < num_inputs:
            return False
    return True


def local_model_config_path(llm: LLMManager, quantization_mode: str | None) -> Path:
    quant_mode = llm._normalized_quantization_mode(quantization_mode)
    quant_tag = quant_mode or "none"
    candidates = [
        Path(llm._local_model_path(quant_mode)) / f"model_{quant_tag}_config.json",
        Path(llm._legacy_named_model_path(quant_mode)) / f"model_{quant_tag}_config.json",
        Path(llm._legacy_local_model_path(quant_mode)) / f"model_{quant_mode or 'None'}_config.json",
    ]
    for candidate in candidates:
        if local_model_config_matches(candidate, llm.config.model_name):
            return candidate
    return candidates[0]


def local_model_config_matches(config_path: Path, expected_model_name: str) -> bool:
    if not config_path.exists():
        return False
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            saved_config = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return False
    return saved_config.get("model_name") == expected_model_name


def load_or_prepare_model(llm: LLMManager, quantization_mode: str | None, save_local_artifacts: bool) -> None:
    config_path = local_model_config_path(llm, quantization_mode)
    if local_model_config_matches(config_path, llm.config.model_name):
        print(f"Loading local quantized model: {config_path.parent}")
        llm.load_model(quantization_mode=quantization_mode or "None", from_hub=False)
        return

    print(f"Local model not found; loading from Hugging Face Hub")
    llm.load_model(quantization_mode=quantization_mode or "None", from_hub=True)
    if save_local_artifacts:
        print(f"Saving local quantized model to {config_path.parent}")
        llm.save_model()
    else:
        print("Skipping local model save because --no-save-local-artifacts is set.")


def run_model_experiments(
    config: RuntimeConfig,
    queries: Sequence,
    embedder: EmbeddingManager,
    embedded_documents: list,
) -> None:
    system_formats = normalize_formats(config.formats)
    num_inputs = len(queries)
    # One checker shared across models: LettuceDetect lazy-loads once instead of once per model
    checker = HallucinationChecker() if config.hallucination_checker else None
    grounding_modes = global_config.LettuceDetect.resolve_modes(config.grounding_modes)
    if checker is not None:
        print(f"Grounding modes: {', '.join(grounding_modes)}")

    try:
        for model_key in config.models:
            print(f"\n{'=' * 50}\nRunning model: {model_key}\n{'=' * 50}")
            # Check logged rows before downloading, so a fully logged model is skipped without
            # fetching its weights. Not done with --overwrite.
            if not config.overwrite and model_outputs_complete(config, model_key, num_inputs, system_formats):
                print(f"Model {model_key} already fully logged ({num_inputs} rows per pipeline); skipping download and run.")
                continue

            retriever = llm = pipeline = experimenter = orderer = prompt_manager = None
            model_name = None
            completed = False

            try:
                retriever = RetrievalManager(embedded_documents)
                llm_config = LLMConfig(model_key=model_key, quantization_mode=config.quantization)
                if config.max_new_tokens is not None:
                    llm_config.max_new_tokens = config.max_new_tokens
                model_name = llm_config.model_name
                prompt_manager = PromptManager(prompt_profile=llm_config.prompt_profile)
                llm = LLMManager(config=llm_config)
                load_or_prepare_model(llm, config.quantization, config.save_local_artifacts)
                orderer = OrderManager(prompt_manager=prompt_manager, llm_manager=llm)
                pipeline = PipelineManager(
                    config=PipelineConfig(
                        embedding_model=embedder,
                        retrieval_model=retriever,
                        order_model=orderer,
                        prompt_template=prompt_manager,
                        llm_model=llm,
                    )
                )
                pipeline.setup_pipeline()
                experimenter = ExperimentManager(
                    input=list(queries),
                    evaluation_manager=EvaluationManager(pipeline=pipeline),
                )

                global_config.General.LOCAL_OUTPUTS_PATH = str(model_output_path(config.output_root, model_key))
                experimenter.multi_passthrough(
                    config.retrievals,
                    config.orders,
                    config.definitions,
                    system_formats,
                    overwrite=config.overwrite,
                    hallucination_checker=checker,
                    grounding_modes=grounding_modes,
                    per_row_seed=config.per_row_seed,
                    batch_size=config.batch_size,
                )

                if config.zero_shot:
                    experimenter.multi_passthrough(
                        retrievals=["none"],
                        orders=["none"],
                        definition_types=[-1],
                        system_formats=[False],
                        overwrite=config.overwrite,
                        per_row_seed=config.per_row_seed,
                        batch_size=config.batch_size,
                    )
                completed = True
            finally:
                del retriever, llm, pipeline, experimenter, orderer, prompt_manager
                memclear(True)

            if completed and config.purge_model_cache and model_name is not None:
                purge_model_artifacts(config, model_name, config.quantization)

            print(f"Model {model_key} done, memory cleared.")
    finally:
        if checker is not None:
            checker.unload()


def run_knn_baseline(
    config: RuntimeConfig,
    queries: Sequence,
    embedder: EmbeddingManager,
    embedded_documents: list,
) -> None:
    print("Embedding all queries for kNN baseline...")
    embeddings = embedder.batch_embed_texts([doc.content for doc in queries])
    labeler = KNNLabeler(
        RetrievalManager(embedded_documents),
        embeddings,
        k=config.knn_k,
    )
    labeler.run_experiment(
        queries=queries,
        retrieval_strategies=config.retrievals,
        output_dir=config.knn_output_dir,
        overwrite=config.overwrite,
    )


def normalize_formats(formats: Iterable[str]) -> list[bool]:
    ordered_formats = []
    for item in formats:
        value = item == "old"
        if value not in ordered_formats:
            ordered_formats.append(value)
    return ordered_formats


# Canonical FRAUD-ICL condition for the contamination probe. Fixed, not configurable: the probe
# compares the paper's headline retrieval condition against zero-shot, so varying it would
# measure something other than the paper's retrieval-over-zero-shot gain.
CONTAMINATION_DEFINITION = 1
CONTAMINATION_FORMAT = "new"
CONTAMINATION_RETRIEVAL = "cosine"
CONTAMINATION_ORDER = "in_order"


def run_contamination_check(config: RuntimeConfig, args: argparse.Namespace) -> None:
    """Run the data-contamination probe (App. 'Data Contamination Check').

    Builds a class-stratified subset of the evaluation set plus a label-preserving perturbed
    copy, then runs the canonical FRAUD-ICL condition and zero-shot over both arms. Surface
    perturbation breaks verbatim memorized matches while keeping the deception label, so a
    retrieval gain that survives it is not purely memorization-surfacing.

    Everything is written under args.contamination_root, in its own {original,perturbed}
    subtrees, so the probe never overwrites the main experiment's outputs. Generation is
    resumable: already-logged rows are skipped, as in a normal run.

    Score these outputs with the same parse-error recovery as the main results
    (fraudicl.recovery).
    """
    from fraudicl.datamanager import ContaminationSubsetBuilder

    root = args.contamination_root
    root.mkdir(parents=True, exist_ok=True)
    original_csv = root / "samples_original.csv"
    perturbed_csv = root / "samples_perturbed.csv"

    print("=" * 60)
    print("DATA CONTAMINATION CHECK")
    print("=" * 60)

    builder = ContaminationSubsetBuilder(seed_value=args.contamination_seed)
    builder.build(
        full_samples_path=config.samples_path,
        original_path=original_csv,
        perturbed_path=perturbed_csv,
        subset_n=args.contamination_n,
    )

    for arm, samples_path in (("original", original_csv), ("perturbed", perturbed_csv)):
        arm_root = root / arm
        arm_root.mkdir(parents=True, exist_ok=True)
        print("\n" + "=" * 60)
        print(f"CONTAMINATION ARM: {arm}")
        print("=" * 60)

        # The probe scores labels only, so the grounding checker is disabled: it would add a
        # LettuceDetect forward pass per row for a metric this analysis never reads.
        arm_config = replace(
            config,
            samples_path=samples_path,
            output_root=arm_root,
            processing_output_dir=None,
            definitions=[CONTAMINATION_DEFINITION],
            formats=[CONTAMINATION_FORMAT],
            retrievals=[CONTAMINATION_RETRIEVAL],
            orders=[CONTAMINATION_ORDER],
            hallucination_checker=False,
            zero_shot=True,
            run_knn=False,
            knn_output_dir=arm_root / "kNN-Generation",
        )
        apply_global_paths(arm_config)
        queries = load_queries(arm_config)
        embedder, embedded_documents = setup_embedding_and_store(arm_config)
        run_model_experiments(
            config=arm_config,
            queries=queries,
            embedder=embedder,
            embedded_documents=embedded_documents,
        )
        print(f"DONE: {samples_path.name} -> {arm_root}")

    print("\n" + "=" * 60)
    print(f"Contamination check complete. Outputs under {root}")
    print("Score both arms with fraudicl.recovery, as for the main grid.")
    print("=" * 60)


def main() -> None:
    config, args = parse_args()
    apply_global_paths(config)
    try:
        if args.contamination_check:
            run_contamination_check(config, args)
            return

        queries = load_queries(config)
        embedder, embedded_documents = setup_embedding_and_store(config)

        run_model_experiments(
            config=config,
            queries=queries,
            embedder=embedder,
            embedded_documents=embedded_documents,
        )

        if config.run_knn:
            run_knn_baseline(
                config=config,
                queries=queries,
                embedder=embedder,
                embedded_documents=embedded_documents,
            )
    finally:
        if config.cleanup_download_cache:
            cleanup_download_cache(config)


def cleanup_download_cache(config: RuntimeConfig) -> None:
    cache_dir = hf_cache_path(config)
    if not cache_dir.exists():
        return
    print(f"Cleaning Hugging Face download cache: {cache_dir}")
    shutil.rmtree(cache_dir)


def purge_model_artifacts(config: RuntimeConfig, model_name: str, quantization: str | None) -> None:
    """Delete a finished model's HF download cache and local quantized snapshot.

    Only model-specific paths are removed; shared artifacts (embedder,
    LettuceDetect, document store) stay so the remaining models can reuse them.
    """
    quant_tag = quantization or "none"
    model_tag = model_name.replace("/", "_")
    hub_tag = f"models--{model_name.replace('/', '--')}"
    cache_dir = hf_cache_path(config)
    candidates = [
        saved_path(config) / "models" / f"{model_tag}_model_{quant_tag}",
        saved_path(config) / f"{model_tag}_model_{quant_tag}",
        cache_dir / "hub" / hub_tag,
        cache_dir / "transformers" / hub_tag,
    ]
    for path in candidates:
        if path.exists():
            print(f"Purging finished-model artifacts: {path}")
            shutil.rmtree(path)
    # The Xet chunk cache is content-addressed (not named per model), so it cannot be
    # purged selectively; clear the whole pinned Xet dir between models to reclaim its
    # share of the disk. Harmless when Xet is disabled (the dir simply won't exist).
    xet_cache = cache_dir / "xet"
    if xet_cache.exists():
        print(f"Purging Xet chunk cache: {xet_cache}")
        shutil.rmtree(xet_cache)


if __name__ == "__main__":
    main()
