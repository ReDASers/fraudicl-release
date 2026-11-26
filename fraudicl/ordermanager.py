import fraudicl.global_config as global_config
from fraudicl.helper import session_timer
from fraudicl.promptmanager import PromptManager
from fraudicl.llmmanager import LLMManager
from typing import List, Dict, Any, Tuple, Optional, Union
from haystack import component, default_to_dict, default_from_dict
from random import seed, shuffle, sample, randint
from haystack.dataclasses import Document
from math import factorial
import numpy as np

@component
class OrderManager:
    """Haystack component for ordering k-shot examples using various strategies (DEmO, random, etc.)."""

    def __init__(self,
                 prompt_manager: Optional[PromptManager] = None,
                 llm_manager: Optional[LLMManager] = None
                 ) -> None:
        """Initialize with prompt and LLM managers for DEmO ordering strategy."""
        self.llm_manager = llm_manager or LLMManager()
        if prompt_manager is None:
            prompt_profile = self.llm_manager.config.prompt_profile
            self.prompt_manager = PromptManager(prompt_profile=prompt_profile)
        else:
            self.prompt_manager = prompt_manager
        seed(global_config.General.RANDOM_SEED)

    @component.output_types(
        documents=List[Document], 
        indices=List[int]
    )
    def run(self, documents: List[Document], order_strategy: str, definition_type: int, use_legacy_format: Union[bool, int], test_text: str) -> Dict[str, Any]:
        """Apply ordering strategy to documents; returns ordered docs and original indices."""
        method = getattr(self, order_strategy)
        result = method(documents, definition_type, use_legacy_format, test_text)
        return {"documents": result[0], "indices": result[1]}

    def none(self, retrieved_documents: List[Document], definition_type: int, use_legacy_format: Union[bool, int], test_text: str) -> Tuple[List[Document], List[int]]:
        """Zero-shot baseline: pass through empty document list unchanged."""
        return ([], [])

    def in_order(self, retrieved_documents: List[Document], definition_type: int, use_legacy_format: Union[bool, int], test_text: str) -> Tuple[List[Document], List[int]]:
        """Return documents in original retrieval order (baseline)."""
        indices = list(range(len(retrieved_documents)))
        return (retrieved_documents, indices)

    def random_order(self, retrieved_documents: List[Document], definition_type: int, use_legacy_format: Union[bool, int], test_text: str) -> Tuple[List[Document], List[int]]:
        """Shuffle documents randomly (control baseline)."""
        indices = list(range(len(retrieved_documents)))
        shuffled_indices = indices.copy()
        shuffle(shuffled_indices)
        ordered_docs = [retrieved_documents[i] for i in shuffled_indices]
        return (ordered_docs, shuffled_indices)
    
    def reverse_order(self, retrieved_documents: List[Document], definition_type: int, use_legacy_format: Union[bool, int], test_text: str) -> Tuple[List[Document], List[int]]:
        """Reverse retrieval order (recency bias baseline)."""
        indices = list(range(len(retrieved_documents)))
        reversed_indices = indices[::-1]
        ordered_docs = [retrieved_documents[i] for i in reversed_indices]
        return (ordered_docs, reversed_indices)
    
    def demo_order(self, retrieved_documents: List[Document], definition_type: int, use_legacy_format: Union[bool, int], test_text: str) -> Tuple[List[Document], List[int]]:
        """Apply DEmO algorithm: entropy-based candidate selection + influence maximization."""
        possible_labels = list(global_config.Database.LABEL_MAPPING.values())[:2] # deceptive, not deceptive are the only possible labels in DiFrauD
        top_candidates = self._calculate_entropy(retrieved_documents, definition_type, use_legacy_format, possible_labels)
        ordered_docs = self._calculate_influence(top_candidates, test_text, definition_type, use_legacy_format, possible_labels)
        indices = [retrieved_documents.index(d) for d in ordered_docs]
        return (ordered_docs, indices)
    
    def _sample_permutations(self, documents: List[Document], num_samples: int) -> List[List[Document]]:
        """Sample permutations lazily without materializing all permutations.

        Uses Fisher-Yates shuffle to generate unique random permutations.
        Falls back to generating all permutations only if num_samples >= total permutations.
        """
        n = len(documents)
        total_perms = factorial(n)

        # If we need most/all permutations, just generate them all
        if num_samples >= total_perms:
            import itertools
            all_perms = [list(p) for p in itertools.permutations(documents)]
            shuffle(all_perms)
            return all_perms[:num_samples]

        # Otherwise, sample unique permutations lazily
        seen_indices = set()
        sampled_perms = []
        indices = list(range(n))

        max_attempts = num_samples * global_config.Runtime.PERM_SAMPLE_MULTIPLIER
        attempts = 0

        while len(sampled_perms) < num_samples and attempts < max_attempts:
            # Generate a random permutation using Fisher-Yates
            perm_indices = indices.copy()
            shuffle(perm_indices)
            perm_key = tuple(perm_indices)

            if perm_key not in seen_indices:
                seen_indices.add(perm_key)
                sampled_perms.append([documents[i] for i in perm_indices])

            attempts += 1

        return sampled_perms

    def _calculate_entropy(self, documents: List[Document], definition_type: int, use_legacy_format: Union[bool, int], possible_labels: List[str]) -> List[Tuple[List[Document], float, List[float]]]:
        """Calculate entropy of the documents."""
        # Get n samples (or pi permutations) using lazy sampling
        num_permutations = global_config.Order.PI_SAMPLE
        total_possible = factorial(len(documents))

        if num_permutations > total_possible:
            print(f"Requested {num_permutations} sample permutations but only have {total_possible} total permutations.")
            num_permutations = total_possible

        subset_permutations = self._sample_permutations(documents, num_permutations)

        # Instantiate results and values
        perm_entropy = []
        content_free_tokens = global_config.Order.CONTENT_FREE_TOKENS

        # Compute entropy
        for perm in subset_permutations:
            entropy, avg_probs = self._compute_content_free_entropy(perm, possible_labels, content_free_tokens, definition_type, use_legacy_format)
            perm_entropy.append((perm, entropy, avg_probs))

        # Keep top-k permutations by entropy
        num_k_candidates = global_config.Order.TOP_K_ENTROPY
        if num_k_candidates > num_permutations:
            print(f"Requested {num_k_candidates} top candidates kept from entropy filtering but only have {num_permutations} sampled permutations.")
            num_k_candidates = num_permutations
        top_candidates = sorted(perm_entropy, key=lambda x: x[1], reverse=True)[:num_k_candidates]

        return top_candidates

    def _compute_content_free_entropy(
        self,
        perm: List[Document],
        possible_labels: List[str],
        content_free_tokens,
        definition_type: int,
        use_legacy_format: Union[bool, int]
    ) -> Tuple[float, List[float]]:
        # Instantiate results
        probs_list = []

        # Get probabilities for each content-free token
        for token in content_free_tokens:
            prompt = self.prompt_manager.run(
                definition_type=definition_type,
                use_legacy_format=use_legacy_format,
                top_k_examples=perm,
                text=token
            )["prompt"]
            probs = self.llm_manager._predict_label_probs(prompt)
            probs_list.append([probs[label] for label in possible_labels])
        avg_probs = np.mean(probs_list, axis=0)

        # Calculate entropy
        entropy = -np.sum([p * np.log(p + global_config.Runtime.LOG_EPSILON) for p in avg_probs if p > 0])
        return entropy, avg_probs

    def _calculate_influence(
        self,
        top_candidates: List[Tuple[List[Document], float, List[float]]],
        test_text: str,
        definition_type: int,
        use_legacy_format: Union[bool, int],
        possible_labels: List[str]
    ) -> List[Document]:
        """Return best order based on influence."""
        best_influence = -float('inf')
        ordered_docs = None

        for perm, entropy, avg_content_free_probs in top_candidates:
            # Build prompt for this test instance
            prompt = self.prompt_manager.run(
                definition_type=definition_type,
                use_legacy_format=use_legacy_format,
                top_k_examples=perm,
                text=test_text
            )["prompt"]

            # Get label probabilities for real input
            probs = self.llm_manager._predict_label_probs(prompt)
            probs_list = [probs[label] for label in possible_labels]
            l_idx = np.argmax(probs_list)
            influence = probs_list[l_idx] - avg_content_free_probs[l_idx] # difference between real and content-free probability for predicted label

            if influence > best_influence:
                best_influence = influence
                ordered_docs = perm

        return ordered_docs

    def to_dict(self) -> Dict[str, Any]:
        """Serialize component to dictionary."""
        return default_to_dict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OrderManager":
        """Deserialize component from dictionary."""
        return default_from_dict(cls, data)
