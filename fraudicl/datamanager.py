import fraudicl.global_config as global_config
from fraudicl.helper import session_timer
import csv
import os
import random as random_module
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any
import numpy as np
import pandas as pd
from datasets import load_dataset
from haystack.dataclasses import Document
from huggingface_hub import hf_hub_download
from random import seed, sample

@dataclass
class DataConfig:
    """Configuration for dataset processing."""
    dataset_path: str = global_config.General.LOCAL_DIFRAUD_PATH
    hf_revision: str = "main"
    local_dataset_path: Optional[str] = None
    save_downloads: bool = True

    # Keeping only documents with n words, where MIN_WORDS <= n <= MAX_WORDS
    max_words: int = global_config.Database.MAX_WORDS
    min_words: int = global_config.Database.MIN_WORDS

    # Determines whether domain stats should be reported
    stats: bool = True

    processing_output_path: Optional[str] = None

class DataManager:
    """Main processor for DiFrauD dataset."""
    
    def __init__(self, config: Optional[DataConfig] = None) -> None:
        """
        Instance attributes:
            config              contains default values for data pre-processing
            train               stores the pre-processed training documents
            validation          stores the pre-processed validation documents
            test                stores the pre-processed testing documents
        """
        self.config = config or DataConfig()
        self.train = None
        self.validation = None
        self.test = None
        self.length_by_domain = {}

    def process_all_domains(self, split: str = "train") -> None:
        """Process all domains and return combined results."""
        print("Loading and converting domains from DiFrauD...")
        print("="*50)
        combined_results = []
        successful_domains = 0
        failed_domains = 0
        processing_path = self.config.processing_output_path or global_config.General.LOCAL_PROCESSING_PATH
        filepath = f"{processing_path}/{split}_failed_processing.csv"
        os.makedirs(processing_path, exist_ok=True)
        with open(filepath, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["domain", "text", "label", "error", "truncated"])
            writer.writeheader()
        session_timer.start_time(f"processing-ALL")
        for domain_name in global_config.Database.DOMAIN_MAPPING.keys():
            try:
                domain_results = self._process_domain(domain_name, split)
                combined_results.extend(domain_results) # merge results
                successful_domains += 1
            except Exception as e:
                print(f"Failed to process domain {domain_name}: {e}")
                failed_domains += 1
                continue
        setattr(self, split, combined_results)
        if not combined_results:
            raise RuntimeError(
                f"No documents were loaded for split '{split}'. "
                f"Check that DataConfig.dataset_path points to a valid DiFrauD dataset. "
                f"Current path: {self.config.dataset_path}"
            )
        session_timer.stop_time(f"processing-ALL")
        print("="*50)
        print(f"Processing complete: {successful_domains} successful, {failed_domains} failed")
        processed, skipped, truncated = self._report_meta_stats()
        print(f"Total processed: {processed}, total skipped: {skipped}, total truncated: {truncated}")
        session_timer.report_time(f"processing-ALL")
        return

    def _process_domain(self, domain_name: str, split: str = "train") -> list:
        """Process all examples in a single domain."""
        print("-"*30)
        print(f"Processing domain: {domain_name}")
        results = []
        if self.config.stats:
            text_lengths = [] # track lengths during processing
        processed_count, skipped_count, trunc_count = 0, 0, 0
        filepath = f"{self.config.processing_output_path or global_config.General.LOCAL_PROCESSING_PATH}/{split}_failed_processing.csv"
        session_timer.start_time(f"processing-{domain_name}")
        data = self._load_domain_dataset(domain_name, split)
        domain_tag = global_config.Database.DOMAIN_MAPPING[domain_name]
        for raw_document in data: # train, test, validation
            try:
                problem = False
                document, word_len, trunc = self._process_document(raw_document, domain_tag)                    
                if document.content is not None:
                    processed_count += 1
                    results.append(document)
                    if trunc:
                        trunc_count += 1
                else:
                    skipped_count += 1
                    problem = True

                if problem:
                    error = f"Text under {self.config.min_words} words"
                else:
                    error = None

                if trunc or problem:
                    self._append_failed_document(
                        filepath=filepath,
                        domain=domain_tag,
                        raw_document=raw_document, 
                        error=error, 
                        trunc=trunc
                    )
                
                if self.config.stats:
                    text_lengths.append(word_len)

            except Exception as e:
                skipped_count += 1
                print(f"Error processing example: {e}")
                if filepath:
                    self._append_failed_document(
                        filepath=filepath, 
                        domain=domain_tag, 
                        raw_document=raw_document, 
                        error=str(e), 
                        trunc=False
                    )
        
        self.length_by_domain[domain_tag] = [processed_count, skipped_count, trunc_count]
        session_timer.stop_time(f"processing-{domain_name}")
        print(f"Processed {processed_count} examples, skipped {skipped_count}")
        if self.config.stats: 
            self._report_stats(text_lengths)
        session_timer.report_time(f"processing-{domain_name}")
        print("")
        
        return results

    def _load_domain_dataset(self, domain_name: str, split: str):
        """Load one DiFrauD domain from Hugging Face or a compatible local dataset path."""
        if self.config.dataset_path == "redasers/difraud":
            local_file = self._local_difraud_file(domain_name, split)
            if local_file.exists():
                print(f"Loading cached DiFrauD split from {local_file}")
                return load_dataset("json", data_files=str(local_file), split="train")

            data_file = hf_hub_download(
                repo_id=self.config.dataset_path,
                filename=f"{domain_name}/{split}.jsonl",
                repo_type="dataset",
                revision=self.config.hf_revision,
            )
            if self.config.save_downloads:
                local_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(data_file, local_file)
                data_file = str(local_file)
            return load_dataset("json", data_files=data_file, split="train")

        return load_dataset(
            self.config.dataset_path,
            domain_name,
            split=split,
            revision=self.config.hf_revision,
        )

    def _local_difraud_file(self, domain_name: str, split: str) -> Path:
        base_path = self.config.local_dataset_path
        if base_path is None:
            dataset_tag = self.config.dataset_path.replace("/", "_")
            base_path = f"{global_config.General.LOCAL_SAVED_PATH}/datasets/{dataset_tag}/{self.config.hf_revision}"
        return Path(base_path) / domain_name / f"{split}.jsonl"

    def _process_document(self, difraud_entry: Dict, domain: str) -> Tuple[Document, int, bool]:
        """Processes textual document into Haystack Document."""
        words = difraud_entry["text"].split()
        word_len = len(words)
        trunc = True if word_len > self.config.max_words else False
        if word_len >= self.config.min_words:
            processed_text = " ".join(words[:self.config.max_words])
        else:
            processed_text = None
        label = difraud_entry["label"]
        tag = self._convert_label_to_tag(label)
        document = Document(
            content=processed_text,
            meta={"domain": domain, "label": label, "tag": tag}
        )
        return [document, word_len, trunc]

    def _convert_label_to_tag(self, label: int) -> str:
        """Convert binary label to descriptive tag."""
        if label not in global_config.Database.LABEL_MAPPING:
            raise ValueError(f"Invalid label: {label}. Expected {global_config.Database.LABEL_MAPPING.keys()}")
        else:
            return global_config.Database.LABEL_MAPPING[label]

    def _append_failed_document(self, filepath: str, domain: str, raw_document: Dict, error: Union[str, None], trunc: Union[bool, int, None]) -> None:
        """Append a failed document to the CSV."""
        try:
            with open(filepath, mode="a", newline="", encoding="utf-8") as csvfile:
                writer = csv.DictWriter(
                    csvfile, 
                    fieldnames=["domain", "text", "label", "error", "truncated"],
                    quoting=csv.QUOTE_ALL
                )
                writer.writerow({
                    "domain": domain,
                    "text": raw_document.get("text", "").strip()[:global_config.Runtime.CSV_CELL_CHAR_LIMIT],
                    "label": raw_document.get("label", ""),
                    "error": error,
                    "truncated": trunc
                })
        except Exception as write_err:
            print(f"Failed to write document to CSV: {write_err}")

    def _report_stats(self, text_lengths: list[int]) -> None:
        """Calculate and report text length statistics for given domain."""
        avg_length = np.mean(text_lengths)
        max_length = np.max(text_lengths)
        min_length = np.min(text_lengths)
        print(f"Avg: {avg_length:.1f}, Max: {max_length}, Min: {min_length} words")

    def _report_meta_stats(self) -> Tuple[int, int, int]:
        """Calculate and report total processed, skipped, and truncated examples."""
        v1_sum, v2_sum, v3_sum = 0, 0, 0
        for values in self.length_by_domain.values():
            v1_sum += values[0]
            v2_sum += values[1]
            v3_sum += values[2]

        return v1_sum, v2_sum, v3_sum
    
class DataSampler:
    """Handles sampling operations on user input documents."""
    
    def __init__(self, data_manager: DataManager, split: str = "validation") -> None:
        """
        Initialize the datasampler with user input documents.
        
        Instance attributes:
            data_manager        stores the data manager containing DiFrauD data
            domain_groups       stores documents grouped by domain for efficient access
            random_seed         stores the random seed for reproducibility
        """
        self.data_manager = data_manager
        self.split = split
        self.documents = getattr(data_manager, split, None)
        self.domain_groups = list(global_config.Database.DOMAIN_MAPPING.keys())
        self.documents_by_domain = None

        if not self.documents:
            raise ValueError(
                f"DataSampler received an empty '{split}' split. "
                f"Run DataManager.process_all_domains('{split}') successfully before sampling."
            )

        self.drawn_ids: set = set()

        # Tracks metadata for the most recent sample() call, used by export_to_csv
        self.last_sample_mode: Optional[str] = None
        self.last_total_samples: Optional[int] = None
        self.last_random_seed: Optional[int] = None

        # Setup
        self.documents_by_domain = self._group_documents_by_domain()

    def _group_documents_by_domain(self) -> Dict[str, List[Document]]:
        """Group selected split documents by domain."""
        # Mapping for domains
        grouped = {}
        for domain in self.domain_groups:
            grouped[domain] = {
                "documents": [],
                "deceptive": 0,
                "non-deceptive": 0, 
                "other": 0
            }
        reverse_mapping = {v: k for k, v in global_config.Database.DOMAIN_MAPPING.items()}

        # Domain creation process
        for doc in self.documents:
            domain_value = doc.meta.get("domain", None)
            domain_label = doc.meta.get("label", None)
            actual_domain = reverse_mapping.get(domain_value, None)
            if actual_domain in grouped:
                grouped[actual_domain]["documents"].append(doc)
            if domain_label == 1:
                grouped[actual_domain]["deceptive"] += 1
            elif domain_label == 0:
                grouped[actual_domain]["non-deceptive"] += 1
            else:
                grouped[actual_domain]["other"] += 1

        return grouped

    def exclude_documents(self, documents: List[Document]) -> None:
        """Mark documents as already drawn so they are excluded from future samples."""
        self.drawn_ids.update(doc.id for doc in documents)

    def _filter_available(self, documents: List[Document]) -> List[Document]:
        """Return only documents whose IDs are not in drawn_ids."""
        if not self.drawn_ids:
            return documents
        return [doc for doc in documents if doc.id not in self.drawn_ids]

    def sample(self, mode: str, total_samples: Union[None, int], random: bool) -> List[Document]:
        """Sample documents from user inputs based on input parameters."""
        if random:
            seed(global_config.General.RANDOM_SEED)
            
        sample_mode_dict = {
            "none": self._no_random_no_strata,
            "all": self._random_no_strata,
            "equal": self._equal_random_strata,
            "prop": self._proportionate_random_strata,
            "class": self._proportionate_random_class_strata
        }
        result = sample_mode_dict[mode](mode, total_samples)

        self.last_sample_mode = mode
        self.last_total_samples = total_samples
        self.last_random_seed = global_config.General.RANDOM_SEED if random else None

        return result

    @staticmethod
    def _allocate_proportionate_counts(strata: List[Dict[str, Any]], total_samples: int) -> List[int]:
        """Allocate exactly total_samples across strata using largest remainders.

        Each stratum dict must include "size". Empty strata receive zero samples.
        Counts are capped at the available stratum size.
        """
        if total_samples < 0:
            raise ValueError("total_samples must be non-negative.")
        total_available = sum(stratum["size"] for stratum in strata)
        if total_samples > total_available:
            raise ValueError(f"Requested {total_samples} samples but only {total_available} documents are available.")
        if total_samples == 0 or total_available == 0:
            return [0 for _ in strata]

        allocations = []
        for index, stratum in enumerate(strata):
            size = stratum["size"]
            exact = (size / total_available) * total_samples if size else 0
            base = min(size, int(exact))
            allocations.append({
                "index": index,
                "count": base,
                "remainder": exact - base,
                "size": size
            })

        remaining = total_samples - sum(item["count"] for item in allocations)
        while remaining > 0:
            candidates = [item for item in allocations if item["count"] < item["size"]]
            if not candidates:
                break
            candidates.sort(key=lambda item: (item["remainder"], item["size"]), reverse=True)
            candidates[0]["count"] += 1
            candidates[0]["remainder"] = 0
            remaining -= 1

        return [item["count"] for item in sorted(allocations, key=lambda item: item["index"])]

    def _no_random_no_strata(self, mode: str, total_samples: Union[None, int]) -> List[Document]:
        """Return the first given number of samples without domain considerations (no strata)."""
        available = self._filter_available(self.documents)
        if total_samples is None:
            print("-"*30)
            print(f"No sampling performed: returning all samples from {self.split} dataset")
            self.drawn_ids.update(doc.id for doc in available)
            print(f"Total documents: {len(available)}")
            return available
        else:
            print("-"*30)
            print(f"No random sampling performed: returning first {total_samples} samples from {self.split} dataset")
            result = available[:total_samples]
            self.drawn_ids.update(doc.id for doc in result)
            print(f"Total documents: {len(result)}")
            return result

    def _random_no_strata(self, mode: str, total_samples: Union[None, int]) -> List[Document]:
        """Return random sample without domain considerations (no strata)."""
        if total_samples is None:
            raise ValueError(f"Total_samples cannot be None with mode: {mode}.")
        available = self._filter_available(self.documents)
        print("-"*30)
        print(f"Random sampling: {total_samples} samples from {self.split} dataset")
        session_timer.start_time(f"recent-validation-sampling")
        sampled_documents = sample(available, total_samples)
        session_timer.stop_time(f"recent-validation-sampling")
        self.drawn_ids.update(doc.id for doc in sampled_documents)
        print("="*50)
        print("Sampling completed")
        print(f"Total documents: {len(sampled_documents)}")
        session_timer.report_time(f"recent-validation-sampling")
        return sampled_documents
    
    def _equal_random_strata(self, mode: str, total_samples: Union[None, int]) -> List[Document]:
        """Return random sample evenly distributed across domains."""
        if total_samples is None:
            raise ValueError(f"Total_samples cannot be None with mode: {mode}.")
        num_domains = len(self.domain_groups)
        samples_per_domain = total_samples // num_domains
        remaining_samples = total_samples % num_domains
        print("-"*30)
        print(f"Equal access stratified sampling: {samples_per_domain} samples per domain from {total_samples} total samples")
        if remaining_samples > 0:
            print(f"Additional {remaining_samples} samples will be distributed to first {remaining_samples} domain(s)")
        print("")
        final_samples = []
        samples_taken_by_domain = {}
        session_timer.start_time(f"recent-validation-sampling")
        for i, domain in enumerate(self.domain_groups):
            domain_sample_count = samples_per_domain + (1 if i < remaining_samples else 0) # +1 if have leftover samples

            try:
                documents = self._filter_available(self._get_documents_by_domain(domain)["documents"])
            except Exception as e:
                print(f"Skipping domain '{domain}': {e}")
                samples_taken_by_domain[domain] = 0
                continue

            available = len(documents)
            if domain_sample_count > available:
                print(f"Warning: Domain '{domain}' has only {available} documents, requested {domain_sample_count}")
                domain_sample_count = available

            if domain_sample_count > 0:
                sampled = sample(documents, domain_sample_count)
                final_samples.extend(sampled)
                samples_taken_by_domain[domain] = len(sampled)
            else:
                samples_taken_by_domain[domain] = 0

        session_timer.stop_time(f"recent-validation-sampling")
        self.drawn_ids.update(doc.id for doc in final_samples)
        print("Stratified sampling results:")
        for domain, count in samples_taken_by_domain.items():
            print(f"{domain}: {count} samples")
        print(f"Total documents: {total_samples}")
        session_timer.report_time(f"recent-validation-sampling")
        print("="*50)
        print("Sampling completed")
        return final_samples

    def _proportionate_random_strata(self, mode: str, total_samples: Union[None, int]) -> List[Document]:
        """Return random sample proportionate to domain sizes."""
        if total_samples is None:
            raise ValueError(f"Total_samples cannot be None with mode: {mode}.")
        num_domains = len(self.domain_groups)
        strata = []
        for domain in self.domain_groups:
            domain_info = self._get_documents_by_domain(domain)
            documents = self._filter_available(domain_info["documents"])
            strata.append({
                "domain": domain,
                "documents": documents,
                "size": len(documents)
            })
        total_size = sum(stratum["size"] for stratum in strata)
        allocated_counts = self._allocate_proportionate_counts(strata, total_samples)
        final_samples = []
        samples_taken_by_domain = {}
        print("-"*30)
        print(f"Proportionate stratified sampling: {total_samples} samples across {num_domains} domain(s)")
        print("")
        session_timer.start_time(f"recent-validation-sampling")
        for stratum, domain_sample_count in zip(strata, allocated_counts):
            domain = stratum["domain"]
            percentage = stratum["size"] / total_size if total_size else 0
            documents = stratum["documents"]
            if domain_sample_count > 0:
                sampled = sample(documents, domain_sample_count)
                final_samples.extend(sampled)
                samples_taken_by_domain[domain] = [len(sampled), percentage]
            else:
                samples_taken_by_domain[domain] = [0, percentage]

        session_timer.stop_time(f"recent-validation-sampling")
        self.drawn_ids.update(doc.id for doc in final_samples)
        print("Stratified sampling results:")
        for domain, count in samples_taken_by_domain.items():
            print(f"{domain}: {count[0]} samples ({(count[1] * 100):.2f}%)")
        print(f"Total documents: {len(final_samples)}")
        session_timer.report_time(f"recent-validation-sampling")
        print("="*50)
        print("Sampling completed")
        return final_samples

    def _proportionate_random_class_strata(self, mode: str, total_samples: Union[None, int]) -> List[Document]:
        """Return random sample proportionate to class distribution within domains."""
        if total_samples is None:
            raise ValueError(f"Total_samples cannot be None with mode: {mode}.")
        num_domains = len(self.domain_groups)
        domain_classes = ["non-deceptive", "deceptive"]
        strata = []
        for domain in self.domain_groups:
            domain_info = self._get_documents_by_domain(domain)
            documents = domain_info["documents"]
            for label, domain_class in enumerate(domain_classes):
                filtered_docs = [doc for doc in documents if doc.meta.get("label", "") == label]
                available_docs = self._filter_available(filtered_docs)
                strata.append({
                    "domain": domain,
                    "class": domain_class,
                    "documents": available_docs,
                    "size": len(available_docs)
                })
        total_size = sum(stratum["size"] for stratum in strata)
        allocated_counts = self._allocate_proportionate_counts(strata, total_samples)
        final_samples = []
        samples_taken_by_domain = {
            domain: {
                domain_class: [0, 0]
                for domain_class in domain_classes
            }
            for domain in self.domain_groups
        }
        print("-"*30)
        print(f"Proportionate stratified sampling by class: {total_samples} samples across {num_domains} domain(s)")
        print("")
        session_timer.start_time(f"recent-validation-sampling")
        for stratum, stratum_sample_count in zip(strata, allocated_counts):
            domain = stratum["domain"]
            domain_class = stratum["class"]
            domain_class_perc = stratum["size"] / total_size if total_size else 0
            if stratum_sample_count > 0:
                sampled = sample(stratum["documents"], stratum_sample_count)
                final_samples.extend(sampled)
            samples_taken_by_domain[domain][domain_class] = [stratum_sample_count, domain_class_perc]

        session_timer.stop_time(f"recent-validation-sampling")
        self.drawn_ids.update(doc.id for doc in final_samples)
        print("Stratified sampling results: (% = of total dataset)")
        for domain, dict in samples_taken_by_domain.items():
            print(f"{domain}: {domain_classes[0].title()} - {dict[domain_classes[0]][0]} ({(dict[domain_classes[0]][1] * 100):.2f}%), {domain_classes[1].title()} - {dict[domain_classes[1]][0]} ({(dict[domain_classes[1]][1] * 100):.2f}%)")
        print(f"Total documents: {len(final_samples)}")
        session_timer.report_time(f"recent-validation-sampling")
        print("="*50)
        print("Sampling completed")
        return final_samples

    def _get_documents_by_domain(self, domain: str) -> Dict[str, Any]:
        """Given domain, return documents from that domain."""
        if domain not in self.documents_by_domain:
            raise ValueError(f"Domain '{domain}' not found in {self.split} data.")
        return self.documents_by_domain[domain]

    EXPORT_HEADERS = ["id", "content", "domain", "label", "tag", "split", "random_seed", "sample_mode", "total_samples"]

    def export_to_csv(self, documents: List[Document], filepath: str, overwrite: bool = True) -> None:
        """Export sampled documents to CSV, recording the seed and method used to draw them.

        overwrite=True replaces the file (and writes a header); overwrite=False appends
        the documents to an existing file, writing a header only if the file is new.
        """
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        write_header = overwrite or not os.path.exists(filepath)
        write_mode = "w" if overwrite else "a"

        with open(filepath, mode=write_mode, newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.EXPORT_HEADERS, quoting=csv.QUOTE_ALL)
            if write_header:
                writer.writeheader()
            for doc in documents:
                writer.writerow({
                    "id": doc.id,
                    "content": doc.content,
                    "domain": doc.meta.get("domain"),
                    "label": doc.meta.get("label"),
                    "tag": doc.meta.get("tag"),
                    "split": self.split,
                    "random_seed": self.last_random_seed,
                    "sample_mode": self.last_sample_mode,
                    "total_samples": self.last_total_samples,
                })

        action = "Overwrote" if overwrite else "Appended to"
        print(f"{action} {filepath} with {len(documents)} documents "
              f"(mode={self.last_sample_mode}, total_samples={self.last_total_samples}, seed={self.last_random_seed})")

    @staticmethod
    def load_from_csv(filepath: str) -> List[Document]:
        """Reconstruct a list of Documents from a CSV produced by export_to_csv."""
        documents = []
        with open(filepath, mode="r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                label = row.get("label", "")
                documents.append(Document(
                    id=row["id"],
                    content=row["content"],
                    meta={
                        "domain": row.get("domain"),
                        "label": int(label) if label not in (None, "") else None,
                        "tag": row.get("tag"),
                    }
                ))
        return documents


class ContaminationSubsetBuilder:
    """Builds the original/perturbed subset pair for the data-contamination probe.

    Several DiFrauD source corpora predate the open-weight LLMs under evaluation, so a
    retrieval-over-zero-shot gain could in principle reflect the model recognizing memorized
    benchmark strings rather than learning from the retrieved demonstrations. The probe draws a
    small class-stratified subset of the evaluation set and produces a perturbed copy under
    label-preserving surface edits, which break verbatim matches while leaving the deception
    label intact. Re-running the canonical condition and zero-shot on both arms shows whether
    the gain survives.

    Synonym substitution is not used because it can alter the label. The edits here are purely
    orthographic and positional.

    Both CSVs are written in DataSampler.EXPORT_HEADERS order so the pipeline's load_from_csv
    reads them unmodified, and `id`/`label`/`domain` are identical between the two files, so
    rows align 1:1 for the paired comparison.
    """

    SAMPLE_MODE = "contamination_subset"

    _SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")

    def __init__(self, seed_value: int, typo_rate: float = 0.03, case_rate: float = 0.05) -> None:
        self.seed_value = seed_value
        self.typo_rate = typo_rate
        self.case_rate = case_rate
        self._rng = random_module.Random(seed_value)

    def stratified_subset(self, full: "pd.DataFrame", subset_n: int) -> "pd.DataFrame":
        """Proportionate class-stratified draw over (domain, label).

        Every stratum keeps at least one row so rare cells (e.g. deceptive message-board posts)
        survive the draw rather than being rounded out of the subset entirely.
        """
        frac = subset_n / len(full)
        parts = []
        for _, grp in full.groupby(["domain", "label"], sort=True):
            k = max(1, round(len(grp) * frac))
            parts.append(grp.sample(n=min(k, len(grp)), random_state=self.seed_value))
        return pd.concat(parts).sort_values("id").reset_index(drop=True)

    def _typo(self, word: str) -> str:
        """Transpose two adjacent interior characters, or duplicate one: a keyboard-slip typo."""
        if len(word) < 4:
            return word
        i = self._rng.randint(1, len(word) - 2)
        if self._rng.random() < 0.5:
            return word[:i] + word[i + 1] + word[i] + word[i + 2:]
        return word[:i] + word[i] + word[i:]

    @staticmethod
    def _case_jitter(word: str) -> str:
        if len(word) < 2:
            return word
        return word[0].swapcase() + word[1:]

    def perturb(self, text: str) -> str:
        """Apply label-preserving surface edits to one document."""
        if not isinstance(text, str) or not text.strip():
            return text
        # 1) whitespace / punctuation normalization (collapse runs, normalize quotes and dashes)
        t = text.replace("’", "'").replace("‘", "'")
        t = (t.replace("“", '"').replace("”", '"')
              .replace("—", "-").replace("–", "-"))
        t = re.sub(r"\s+", " ", t).strip()
        # 2) cyclic sentence rotation -- breaks a verbatim prefix match, preserves every sentence
        sents = [s for s in self._SENT_SPLIT.split(t) if s]
        if len(sents) >= 3:
            cut = self._rng.randint(1, len(sents) - 1)
            t = " ".join(sents[cut:] + sents[:cut])
        # 3) light per-word typo + casing jitter
        words = t.split(" ")
        for j, w in enumerate(words):
            if not w:
                continue
            r = self._rng.random()
            if r < self.typo_rate:
                words[j] = self._typo(w)
            elif r < self.typo_rate + self.case_rate:
                words[j] = self._case_jitter(w)
        return " ".join(words)

    def _rows_for(self, df: "pd.DataFrame", perturbed: bool) -> List[Dict[str, Any]]:
        rows = []
        for _, r in df.iterrows():
            rows.append({
                "id": r["id"],
                "content": self.perturb(r["content"]) if perturbed else r["content"],
                "domain": r["domain"],
                "label": int(r["label"]),
                "tag": r.get("tag", ""),
                "split": r.get("split", "validation"),
                "random_seed": self.seed_value,
                "sample_mode": self.SAMPLE_MODE,
                "total_samples": len(df),
            })
        return rows

    def write_subset(self, df: "pd.DataFrame", filepath: Path, perturbed: bool) -> bool:
        """Write one arm's CSV. Returns True if written, False if an identical file was kept.

        An existing file is never overwritten: inference may already have been generated
        against it, and silently replacing the inputs would misalign completed outputs with
        the rows they were produced from. The regenerated content is compared byte-for-byte
        and any drift raises rather than clobbering.
        """
        new_rows = self._rows_for(df, perturbed)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        if filepath.exists():
            old = pd.read_csv(filepath)
            new_content = [r["content"] for r in new_rows]
            old_content = old["content"].astype(str).tolist()
            if len(old_content) == len(new_content) and all(
                    a == b for a, b in zip(old_content, new_content)):
                print(f"exists and reproduces byte-identical, keeping: {filepath}")
                return False
            raise RuntimeError(
                f"{filepath} exists but the regenerated content DIFFERS -- refusing to "
                f"overwrite, because inference may already have been generated against the "
                f"old file. Delete {filepath.parent} and re-run to regenerate from scratch.")
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=DataSampler.EXPORT_HEADERS,
                                    quoting=csv.QUOTE_ALL)
            writer.writeheader()
            writer.writerows(new_rows)
        print(f"wrote {len(new_rows)} rows -> {filepath} (perturbed={perturbed})")
        return True

    def build(self, full_samples_path: Path, original_path: Path,
              perturbed_path: Path, subset_n: int) -> "pd.DataFrame":
        """Draw the subset and write both arms. Returns the subset frame."""
        full = pd.read_csv(full_samples_path)
        subset = self.stratified_subset(full, subset_n)
        print(f"full eval set: {len(full)} rows | contamination subset: {len(subset)} rows")
        print(subset.groupby(["domain", "label"]).size())

        # Perturbation draws from one sequential RNG stream, so the output depends on how many
        # times perturb() has been called. The shipped samples_perturbed.csv was produced after
        # one preview call on the first row; this call reproduces it byte for byte.
        preview = self.perturb(subset.iloc[0]["content"])
        print(f"\n--- ORIGINAL ---\n{subset.iloc[0]['content'][:300]}")
        print(f"\n--- PERTURBED ---\n{preview[:300]}\n")

        self.write_subset(subset, original_path, perturbed=False)
        self.write_subset(subset, perturbed_path, perturbed=True)
        orig = pd.read_csv(original_path)
        pert = pd.read_csv(perturbed_path)
        altered = (orig["content"].astype(str).values != pert["content"].astype(str).values).mean()
        print(f"fraction of documents altered: {altered:.3f}")
        return subset
