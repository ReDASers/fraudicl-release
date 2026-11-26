import fraudicl.global_config as global_config
from fraudicl.helper import memclear, session_timer
from fraudicl.datastorage import VectorDBManager
from haystack.dataclasses import Document
from typing import List, Optional, Dict
import numpy as np
from random import shuffle, sample
from copy import deepcopy
from haystack import component, default_to_dict, default_from_dict


@component
class RetrievalManager:
    """Manages retrieval operations from the document store using various strategies."""

    def __init__(self, all_documents: List[Document]) -> None:
        """Initialize with documents and pre-compute embedding matrix for fast retrieval."""
        self.all_documents = all_documents
        # Pre-compute embedding matrix and norms for vectorized cosine similarity
        self._embedding_matrix: Optional[np.ndarray] = None
        self._embedding_norms: Optional[np.ndarray] = None
        self._embedded_doc_indices: np.ndarray = np.asarray([], dtype=np.int64)
        self._label_indices: Dict[int, np.ndarray] = {}
        self._precompute_embeddings()

    def _precompute_embeddings(self) -> None:
        """Pre-compute embedding matrix and norms for O(1) retrieval setup."""
        if not self.all_documents:
            return
        embedded_pairs = [
            (index, doc.embedding)
            for index, doc in enumerate(self.all_documents)
            if doc.embedding is not None
        ]
        embeddings = [embedding for _, embedding in embedded_pairs]
        if embeddings:
            self._embedding_matrix = np.asarray(embeddings, dtype=np.float32)
            self._embedding_norms = np.linalg.norm(self._embedding_matrix, axis=1)
            self._embedded_doc_indices = np.asarray(
                [index for index, _ in embedded_pairs],
                dtype=np.int64
            )
            for label in (0, 1):
                self._label_indices[label] = np.asarray(
                    [
                        matrix_index
                        for matrix_index, doc_index in enumerate(self._embedded_doc_indices)
                        for doc in [self.all_documents[int(doc_index)]]
                        if int(doc.meta.get("label", -1)) == label and doc.embedding is not None
                    ],
                    dtype=np.int64
                )

    @component.output_types(
        documents=List[Document],
        retrieved_documents=List[Document]
    )
    def run(self, user_query: List[float], retrieval_strategy: str, top_k: int) -> Dict[str, List[Document]]:
        """Retrieve top-k documents using specified strategy (cosine, balanced_cosine, random)."""
        method = getattr(self, retrieval_strategy)
        retrieved_docs = method(user_query, top_k)
        # `documents` is consumed by the orderer; `retrieved_documents` is unconnected so it
        # always surfaces in pipeline.run() output, allowing downstream grounding checks.
        return {"documents": retrieved_docs, "retrieved_documents": retrieved_docs}
    
    def _compute_similarities(self, query_embedding: List[float]) -> np.ndarray:
        """Compute cosine similarities using vectorized operations (10-100x faster)."""
        if self._embedding_matrix is None or self._embedding_norms is None:
            return np.asarray([], dtype=np.float32)
        query_vec = np.asarray(query_embedding, dtype=np.float32)
        query_norm = np.linalg.norm(query_vec)

        # Vectorized dot product and normalization
        dot_products = self._embedding_matrix @ query_vec
        similarities = dot_products / (self._embedding_norms * query_norm + global_config.Runtime.EPSILON)
        return similarities

    @staticmethod
    def _top_k_indices(scores: np.ndarray, top_k: int) -> np.ndarray:
        """Return score indices for top_k values without sorting the full array."""
        if top_k <= 0:
            return np.asarray([], dtype=np.int64)
        if top_k >= len(scores):
            return np.argsort(scores)[::-1]
        candidate_indices = np.argpartition(scores, -top_k)[-top_k:]
        return candidate_indices[np.argsort(scores[candidate_indices])[::-1]]

    def cosine(self, query_embedding: List[float], top_k: int) -> List[Document]:
        """Retrieve top-k documents by cosine similarity (vectorized)."""
        sims = self._compute_similarities(query_embedding)

        # Get indices of top_k most similar documents
        top_indices = self._top_k_indices(sims, top_k)
        top_docs = []

        for i in top_indices:
            doc_idx = int(self._embedded_doc_indices[i])
            # Create a copy to avoid mutating the original document
            doc_copy = Document(
                content=self.all_documents[doc_idx].content,
                meta=self.all_documents[doc_idx].meta.copy() if self.all_documents[doc_idx].meta else {},
                embedding=self.all_documents[doc_idx].embedding,
                score=float(sims[i])
            )
            top_docs.append(doc_copy)

        return top_docs

    def balanced_cosine(self, query_embedding: List[float], top_k: int) -> List[Document]:
        """Retrieve top-k documents balanced between labels (vectorized similarity)."""
        # Compute all similarities at once
        sims = self._compute_similarities(query_embedding)

        # Calculate per-label count
        labels = [label for label, indices in self._label_indices.items() if len(indices) > 0]
        num_labels = len(labels)
        if num_labels == 0:
            return self.cosine(query_embedding, top_k)
        per_label = top_k // num_labels
        remainder = top_k % num_labels

        top_docs = []
        # Take per_label from each, and distribute remainder (if any)
        selected_indices = set()
        for label in labels:
            label_indices = self._label_indices[label]
            selected_local = self._top_k_indices(sims[label_indices], per_label)
            for local_idx in selected_local:
                matrix_idx = int(label_indices[local_idx])
                doc_idx = int(self._embedded_doc_indices[matrix_idx])
                selected_indices.add(matrix_idx)
                doc = self.all_documents[doc_idx]
                # Create a copy to avoid mutating the original document
                doc_copy = Document(
                    content=doc.content,
                    meta=doc.meta.copy() if doc.meta else {},
                    embedding=doc.embedding,
                    score=float(sims[matrix_idx])
                )
                top_docs.append(doc_copy)

        # Distribute any remainder to the most similar across all labels
        if remainder > 0:
            remaining_scores = sims.copy()
            if selected_indices:
                remaining_scores[list(selected_indices)] = -np.inf
            for matrix_idx in self._top_k_indices(remaining_scores, remainder):
                doc_idx = int(self._embedded_doc_indices[int(matrix_idx)])
                doc = self.all_documents[doc_idx]
                # Create a copy to avoid mutating the original document
                doc_copy = Document(
                    content=doc.content,
                    meta=doc.meta.copy() if doc.meta else {},
                    embedding=doc.embedding,
                    score=float(sims[matrix_idx])
                )
                top_docs.append(doc_copy)

        return top_docs

    def random(self, query_embedding: List[float], top_k: int) -> List[Document]:
        """Returns top_k random documents from the collection, ignoring the query."""
        if top_k >= len(self.all_documents):
            result = self.all_documents[:]
            shuffle(result)
            return result
        return sample(self.all_documents, top_k)

    def none(self, query_embedding: List[float], top_k: int) -> List[Document]:
        """Zero-shot baseline: return no documents."""
        return []
