import fraudicl.global_config as global_config
from fraudicl.helper import session_timer, memclear
from fraudicl.evaluationmanager import EvaluationManager, HallucinationChecker
import csv
import hashlib
import json
from typing import Dict, List, Hashable, Any, Optional
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
import numpy as np
import pandas as pd
import os
import psutil
import torch
from haystack.dataclasses import Document
from random import seed
from tqdm import tqdm

class ExperimentManager:
    """Manages batch experiments and provides output recording."""
    
    def __init__(self, input: List[Document], evaluation_manager: EvaluationManager) -> None:
        """
        Initialize the ExperimentManager.

        Instance attributes:
            input                       the list of documents to be run through pipeline
            evaluation_manager          the manager that contains all the pipelines
        """
        self.input = input
        self.evaluation_manager = evaluation_manager

        seed(global_config.General.RANDOM_SEED)

    @staticmethod
    def _row_seed(base_seed: int, idx: int, definition_type: int, tag: str, retrieval: str, order: str) -> int:
        """Deterministic per-row seed from the row's identity (not its position in the RNG
        stream), so a resumed row draws exactly what an uninterrupted run would. Uses a stable
        hash (sha256) rather than Python's per-process-salted hash() to stay reproducible across
        processes."""
        key = f"{base_seed}|{idx}|{definition_type}|{tag}|{retrieval}|{order}"
        return int(hashlib.sha256(key.encode()).hexdigest(), 16) % (2 ** 32)

    def multi_passthrough(self, retrievals: List[str], orders: List[str], definition_types: List[int], system_formats: List[bool], overwrite: bool = True, hallucination_checker: Optional[HallucinationChecker] = None, grounding_modes: Optional[List[str]] = None, per_row_seed: bool = False, batch_size: int = 1) -> None:
        """Automate passthroughs through multiple pipelines.

        grounding_modes lists the LettuceDetect reference contexts to score each
        response against ("demos", "demos+query", "demos+query+def"; see
        EvaluationManager.passthrough). Each mode adds a block of grounding
        columns to the log: the primary mode ("demos") keeps the unsuffixed
        names, others are suffixed with the mode slug. Ignored when
        hallucination_checker is None. Defaults to all three modes.

        When per_row_seed is True, the RNG is reseeded per row from the row's identity, making
        random_order/demo_order bitwise-reproducible even across resume (an interrupted run and a
        clean run produce identical rows). When False (default), one seed is set at start-up,
        so resumed stochastic rows are valid but not bitwise-identical to a clean run.

        batch_size > 1 generates that many rows' prompts in one forward pass. Retrieval,
        ordering and prompt assembly stay per row; only generation is batched. This does
        NOT reproduce batch_size=1 on this stack (4-bit weights plus a repetition penalty;
        see --batch-size in main.py and Scripts/verify_batched_generation.py), so treat
        batch_size as part of a run's provenance.
        """
        num_inputs = len(self.input)
        if num_inputs == 0:
            raise ValueError("Experiment input is empty. Check the DataSampler mode and total sample count.")
        # Resolve once: the same mode list drives the CSV header and every row, so
        # header and payload cannot drift apart mid-run.
        grounding_modes = global_config.LettuceDetect.resolve_modes(grounding_modes)
        num_pipelines = len(retrievals) * len(orders) * len(definition_types) * len(system_formats)
        if 0 in definition_types and True in system_formats:
            num_pipelines -= len(retrievals) * len(orders)
        num_passthroughs = num_inputs * num_pipelines
        print(f"Automating {num_inputs} passthroughs/pipeline over {num_pipelines} pipeline(s)")
        print("~"*15)
        print(f"Definitions: {definition_types}")
        print(f"Formats (New/Old): {system_formats}")
        print(f"Retrievals: {retrievals}")
        print(f"Orders: {orders}")
        print("~"*15)
        print(f"Total passthroughs: {num_passthroughs}")
        print("="*50)

        session_timer.start_time(f"multi_passthrough")

        # Pre-embed all query texts in one batch before the pipeline loop
        all_texts = [doc.content for doc in self.input]
        self.evaluation_manager.pipeline.pre_embed(all_texts)

        created_files_this_run = set()
        with tqdm(total=num_pipelines, desc="Pipelines", position=0) as outer_bar:
            for definition_type in definition_types:
                for use_legacy_format in system_formats: # these will just be T/F
                    tag = "old" if use_legacy_format else "new"
                    if definition_type == 0 and use_legacy_format:
                        continue
                    
                    for retrieval in retrievals:
                        for order in orders:
                            print("-"*30)
                            print(f"Pipeline: {retrieval.upper()} - {order.upper()} - DEF {definition_type} - {tag.upper()}")
                            session_timer.start_time(f"passthrough-{order}-{definition_type}-{tag}")
                            # Output file named by definition, format, and retrieval
                            filepath = f"{global_config.General.LOCAL_OUTPUTS_PATH}/def_{definition_type}_{tag}_{retrieval}_outputs.csv"

                            completed_rows = 0
                            if not overwrite:
                                self.create_csv(mode="logging", filepath=filepath, overwrite=False, grounding_modes=grounding_modes)
                                completed_rows = self._count_logged_rows(filepath, order)
                            else:
                                should_overwrite = filepath not in created_files_this_run
                                self.create_csv(mode="logging", filepath=filepath, overwrite=should_overwrite, grounding_modes=grounding_modes)
                                created_files_this_run.add(filepath)

                            if completed_rows >= num_inputs:
                                print(f"All {num_inputs} passthroughs already logged; skipping pipeline.")
                                session_timer.stop_time(f"passthrough-{order}-{definition_type}-{tag}")
                                outer_bar.update(1)
                                continue
                            if completed_rows > 0:
                                if per_row_seed:
                                    print(f"Resuming after {completed_rows} previously logged passthroughs (per-row seeding: bitwise-identical to a clean run).")
                                else:
                                    # Resumed random_order/demo_order rows draw from a fresh RNG
                                    # position, so they are valid but not bitwise-identical to an
                                    # uninterrupted run. Pass --per-row-seed for reproducible resume.
                                    print(f"Resuming after {completed_rows} previously logged passthroughs (random_order/demo_order not bitwise-reproducible; use --per-row-seed).")

                            pending_inputs = self.input[completed_rows:]
                            with open(filepath, "a", newline="", encoding="utf-8") as f:
                                writer = csv.writer(f)
                                with tqdm(pending_inputs, desc=f"{order}-{definition_type}-{tag}", position=1) as inner_bar:
                                    for chunk_start in range(0, len(pending_inputs), batch_size):
                                        chunk = pending_inputs[chunk_start:chunk_start + batch_size]
                                        base_index = completed_rows + chunk_start

                                        def _seed_row(position, _base=base_index):
                                            # Reseed from the row's identity so its random draws
                                            # depend only on (seed, row, combo) -- not on stream
                                            # position -- making resume bitwise-reproducible.
                                            if per_row_seed:
                                                seed(self._row_seed(
                                                    global_config.General.RANDOM_SEED,
                                                    _base + position,
                                                    definition_type, tag, retrieval, order,
                                                ))

                                        session_timer.start_time(f"recent_passthrough")
                                        if batch_size > 1:
                                            batch_results = self.evaluation_manager.passthrough_batch(
                                                texts=[u.content for u in chunk],
                                                true_labels=[u.meta.get("label") for u in chunk],
                                                definition_type=definition_type,
                                                use_legacy_format=use_legacy_format,
                                                order_strategy=order,
                                                retrieval_strategy=retrieval,
                                                hallucination_checker=hallucination_checker,
                                                grounding_modes=grounding_modes,
                                                prepare_hook=_seed_row,
                                            )
                                        else:
                                            _seed_row(0)
                                            batch_results = [self.evaluation_manager.passthrough(
                                                text=chunk[0].content,
                                                true_label=chunk[0].meta.get("label"),
                                                verbose=False,
                                                definition_type=definition_type,
                                                use_legacy_format=use_legacy_format,
                                                order_strategy=order,
                                                retrieval_strategy=retrieval,
                                                hallucination_checker=hallucination_checker,
                                                grounding_modes=grounding_modes
                                            )]
                                        session_timer.stop_time(f"recent_passthrough")
                                        # Per-row elapsed time is no longer separable inside a
                                        # batch, so charge each row an equal share of the call.
                                        batch_elapsed = session_timer.return_elapsed_time("recent_passthrough")
                                        per_row_elapsed = batch_elapsed / max(len(chunk), 1)

                                        i = base_index + len(chunk) - 1
                                        if i % global_config.Runtime.MEMORY_CLEANUP_INTERVAL < len(chunk) and i > 0:
                                            self.report_memory()
                                            memclear(False)

                                        for user_input, results in zip(chunk, batch_results):
                                            writer.writerow(self._build_row(
                                                definition_type,
                                                tag,
                                                retrieval,
                                                order,
                                                results["indices"],
                                                user_input.meta.get("domain"),
                                                user_input.content,
                                                user_input.meta.get("label"),
                                                results["score"],
                                                results["pred"],
                                                results["reasoning"],
                                                results["full_answer"],
                                                results["raw_prompt"],
                                                results["retrieved_examples"],
                                                results["error"],
                                                results,
                                                grounding_modes,
                                                per_row_elapsed
                                            ))
                                            inner_bar.update(1)
                                        # Flush once per batch: a row is only "logged" for resume
                                        # once it reaches disk, and the whole batch is regenerated
                                        # together if the run dies mid-chunk.
                                        f.flush()

                            session_timer.stop_time(f"passthrough-{order}-{definition_type}-{tag}")
                            session_timer.report_time(f"passthrough-{order}-{definition_type}-{tag}")
                            outer_bar.update(1)
                            print("")

        session_timer.stop_time(f"multi_passthrough")
        print("="*50)
        print("Multi-passthrough completed")
        session_timer.report_time(f"multi_passthrough")
        
        return
    
    def _build_row(self, definition, format_tag, retrieval, order, indices, domain, user_input, true_label, score, prediction, reasoning, full_answer, raw_prompt, retrieved_examples, error, grounding, grounding_modes, time) -> list:
        """Build a CSV row for one passthrough result.

        `grounding` is the passthrough result dict; its grounding columns are read
        by name against the same header builder create_csv used, so adding a mode
        can never silently shift columns.
        """
        correct = (score == true_label)
        valid = (score != -1)
        ld = global_config.LettuceDetect
        grounding_cells = [grounding.get(col) for col in ld.columns_for(grounding_modes or [])]
        return [
            definition, format_tag, retrieval, order, indices, domain, user_input,
            true_label, score, prediction, reasoning, valid, correct, full_answer,
            raw_prompt, json.dumps(retrieved_examples, ensure_ascii=False),
            error, *grounding_cells, grounding.get("grounding_modes"), time
        ]
    
    def _count_logged_rows(self, filepath: str, order: str) -> int:
        """Count rows already logged for this order so interrupted runs resume instead of duplicating."""
        if not os.path.exists(filepath):
            return 0
        try:
            logged = pd.read_csv(filepath, usecols=["order"])
        except (pd.errors.EmptyDataError, ValueError):
            return 0
        return int((logged["order"] == order).sum())

    def create_csv(self, mode: str, filepath: str, overwrite: bool = False, grounding_modes: Optional[List[str]] = None) -> None:
        """Create an empty CSV file with headers for logging or summary."""
        
        # Check if file exists and handle overwrite logic
        if os.path.exists(filepath):
            if overwrite:
                os.remove(filepath)
            else:
                # File exists and overwrite is False so do nothing
                return
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Create CSV with headers
        if mode == "logging":
            headers = global_config.Evaluation.logging_headers(grounding_modes)
        elif mode == "domain":
            headers = global_config.Evaluation.DOMAIN_HEADERS
        else:
            raise ValueError(f"CSV headers for {mode} not recognized.")
        
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(headers)

    @staticmethod
    def summarize_overall(df: pd.DataFrame, metric: str) -> pd.DataFrame:
        """Get summary data overall."""
        # Get unique values dynamically
        unique_templates = sorted(df["template"].unique())
        unique_orders = sorted(df["order"].unique())
        unique_domains = sorted(df["domain"].unique())

        # Create all combinations of template and order
        template_order_combinations = []
        for template in unique_templates:
            for order in unique_orders:
                template_order_combinations.append((template, order))

        # Calculate all values for summary
        summary_data = []

        for template, order in template_order_combinations:
            # Skip
            if template == "0_old":
                continue
            
            # Filter data for this template-order combination
            subset = df[(df["template"] == template) & (df["order"] == order)]
        
            if len(subset) == 0:
                continue
            
            # Initialize row data
            defi, frame = template.split("_")
            # "new" is this system; "old" is the RADDICL 1.0 prior-work format kept as a baseline
            framework = "FRAUD-ICL" if frame == "new" else "RADDICL-1.0"
            row_data = {
                "framework": framework,
                "definition": f"Definition {defi}",
                "order strategy": order,
                f"overall {metric}": subset[metric].mean(),
                "overall coverage": subset["coverage"].mean(),
                "avg time": subset["avg_time"].mean()
            }
            # Fill domain-specific values
            for domain in unique_domains:
                domain_data = subset[subset["domain"] == domain]
                if len(domain_data) > 0:
                    row_data[domain] = domain_data[metric].iloc[0]
                else:
                    row_data[domain] = np.nan
            summary_data.append(row_data)

        # Create columns list
        columns = ["framework", "definition", "order strategy", f"overall {metric}", "overall coverage", "avg time"] + unique_domains

        # Create and return summary DataFrame
        df = pd.DataFrame(summary_data, columns=columns)
        return df

    @staticmethod
    def summarize_by_domain(df: pd.DataFrame) -> Dict[Hashable, Any]:
        """Get summary data by domain."""
        count = len(df)
        valid = df["valid"].sum()
        avg_time = df["seconds"].mean()
        coverage = valid / count if count > 0 else 0
        
        # Filter to only valid predictions for accuracy metrics
        valid_group = df[df["valid"] == True]
        
        if len(valid_group) > 0:
            y_true = np.array(valid_group["true_label"])
            y_pred = np.array(valid_group["pred_label"])
            accuracy = np.mean(y_true == y_pred)
            unique_true = np.unique(y_true)
            unique_pred = np.unique(y_pred)
            if len(unique_true) == 1 or len(unique_pred) == 1:
                # All predictions or true labels are the same class
                f1 = precision = recall = accuracy
                if unique_true[0] == 1:
                    tn, fp, fn, tp = 0, 0, len(y_true) - np.sum(y_true == y_pred), np.sum(y_true == y_pred)
                else:
                    tn, fp, fn, tp = np.sum(y_true == y_pred), len(y_true) - np.sum(y_true == y_pred), 0, 0
            else:
                f1 = f1_score(y_true, y_pred, zero_division=0)
                precision = precision_score(y_true, y_pred, zero_division=0)
                recall = recall_score(y_true, y_pred, zero_division=0)
                cm = confusion_matrix(y_true, y_pred, labels=[0,1])
                if cm.shape == (2,2):
                    tn, fp, fn, tp = cm.ravel()
                else:
                    tn = fp = fn = tp = 0
        else:
            accuracy = 0
            f1 = 0
            precision = 0
            recall = 0
            tp = 0
            tn = 0
            fp = 0
            fn = 0

            
        results = {
            "count": count,
            "coverage": coverage,
            "accuracy": accuracy,
            "f1": f1,
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "invalid": count - valid,
            "precision": precision,
            "recall": recall,
            "avg_time": avg_time
        }
        return results

    @staticmethod
    def trim_output(llm_output: Dict[Hashable, Any]) -> Dict[Hashable, Any]:
        """Helper function to trim output."""
        keys_to_remove = ["order", "domain"]
        return {k: v for k, v in llm_output.items() if k not in keys_to_remove}

    @staticmethod
    def report_memory() -> None:
        """Reports on memory statistics."""
        ram_gb = psutil.virtual_memory().used / 1024**3
        ram_pct = psutil.virtual_memory().percent
        
        if torch.cuda.is_available():
            alloc_gb = torch.cuda.memory_allocated() / 1024**3
            max_alloc_gb = torch.cuda.max_memory_allocated() / 1024**3
            reserved_gb = torch.cuda.memory_reserved() / 1024**3
            print(f"Memory: RAM {ram_gb:.1f}GB ({ram_pct:.1f}%) | GPU alloc {alloc_gb:.1f}GB (max {max_alloc_gb:.1f}GB) | GPU reserved {reserved_gb:.1f}GB")
        else:
            print(f"Memory: RAM {ram_gb:.1f}GB ({ram_pct:.1f}%)")
