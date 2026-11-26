import fraudicl.global_config as global_config
from fraudicl.helper import session_timer, determine_device
from dataclasses import dataclass
from typing import Dict, List, Optional
import torch
from haystack.dataclasses import Document
from haystack import component, default_to_dict, default_from_dict
from haystack.components.embedders import SentenceTransformersTextEmbedder, SentenceTransformersDocumentEmbedder
from haystack.utils import ComponentDevice

@dataclass
class EmbeddingConfig:
    """Configuration for embedding model."""
    device_type: str = global_config.General.DEVICE_TYPE
    embedding_model: str = global_config.Embedding.MODEL
    embedding_dimension: int = global_config.Embedding.DIMENSIONS
    normalize_embeddings: bool = global_config.Embedding.NORMALIZE_EMBEDDINGS
    batch_size: int = global_config.Embedding.BATCH_SIZE

class EmbeddingManager:
    """Manages document and text embedding using SentenceTransformers (lazy-loaded)."""

    def __init__(self, config: Optional[EmbeddingConfig] = None) -> None:
        """Initialize with config; embedders are created on first use."""
        self.config = config or EmbeddingConfig()
        self.document_embedder = None
        self.text_embedder = None
        self._batch_cache: Dict[str, List[float]] = {}

    def _setup_document_embedder(self) -> SentenceTransformersDocumentEmbedder:
        """Create document embedder for batch embedding of Document objects."""
        document_embedder = SentenceTransformersDocumentEmbedder(
            model=self.config.embedding_model,
            device=ComponentDevice.from_str(determine_device(self.config.device_type)),
            batch_size=self.config.batch_size,
            normalize_embeddings=self.config.normalize_embeddings,
            progress_bar=True
        )
        document_embedder.warm_up()
        return document_embedder
    
    def _setup_text_embedder(self) -> SentenceTransformersTextEmbedder:
        """Create text embedder for single query string embedding."""
        text_embedder = SentenceTransformersTextEmbedder(
            model=self.config.embedding_model,
            device=ComponentDevice.from_str(determine_device(self.config.device_type)),
            normalize_embeddings=self.config.normalize_embeddings,
            progress_bar=False
        )
        text_embedder.warm_up()
        return text_embedder

    def _embed_documents(self, chunks: List[Document]) -> List[Document]:
        """Embed document chunks and return Documents with embedding field populated."""
        print("-" * 30)
        print(f"Embedding {len(chunks)} chunks")
        session_timer.start_time("embedding-chunks")
        if self.document_embedder is None:
            self.document_embedder = self._setup_document_embedder()
        with torch.inference_mode():
            embedded_result = self.document_embedder.run(chunks)
            embedded_docs = embedded_result["documents"]
        session_timer.stop_time("embedding-chunks")
        print(f"Embedded {len(embedded_docs)} chunks")
        session_timer.report_time("embedding-chunks")
        print("")
        return embedded_docs

    def _embed_texts(self, text: str) -> List[float]:
        """Embed a single text query and return its embedding vector."""
        print("-" * 30)
        print(f"Embedding text query")
        session_timer.start_time("embedding-texts")
        if self.text_embedder is None:
            self.text_embedder = self._setup_text_embedder()
        with torch.inference_mode():
            embedded_result = self.text_embedder.run(text)
            embedded_result = embedded_result["embedding"]
        session_timer.stop_time("embedding-texts")
        print(f"Embedded text query")
        session_timer.report_time("embedding-texts")
        print("")
        return embedded_result

    def batch_embed_texts(self, texts: List[str]) -> Dict[str, List[float]]:
        """Batch-embed a list of query strings; memoized so repeated calls skip the GPU pass."""
        missing = [text for text in dict.fromkeys(texts) if text not in self._batch_cache]
        if missing:
            from sentence_transformers import SentenceTransformer
            device = determine_device(self.config.device_type)
            print(f"Batch embedding {len(missing)} queries...")
            session_timer.start_time("batch-embed-texts")
            model = SentenceTransformer(self.config.embedding_model, device=device)
            with torch.inference_mode():
                embeddings = model.encode(
                    missing,
                    batch_size=self.config.batch_size,
                    show_progress_bar=True,
                    normalize_embeddings=self.config.normalize_embeddings,
                    convert_to_numpy=True
                )
            del model
            session_timer.stop_time("batch-embed-texts")
            session_timer.report_time("batch-embed-texts")
            self._batch_cache.update({text: emb.tolist() for text, emb in zip(missing, embeddings)})
        else:
            print(f"Embedding cache hit for all {len(texts)} queries.")
        return {text: self._batch_cache[text] for text in texts}



@component
class EmbeddingCache:
    """Cache-backed text embedder; does a dict lookup instead of a GPU forward pass."""

    def __init__(self) -> None:
        self._cache: Dict[str, List[float]] = {}

    def populate(self, cache: Dict[str, List[float]]) -> None:
        """Load pre-computed {text: embedding} pairs into the cache.

        Keys are stripped so they match the pipeline lookup, which runs on text.strip()
        (see EvaluationManager._analyze_text). Without this, a query whose source content has
        leading/trailing whitespace would miss the cache and raise KeyError mid-run.
        """
        self._cache = {text.strip(): embedding for text, embedding in cache.items()}

    @component.output_types(embedding=List[float])
    def run(self, text: str) -> Dict[str, List[float]]:
        key = text.strip()
        if key not in self._cache:
            raise KeyError(f"Text not in embedding cache. Call pre_embed() before running the pipeline.")
        return {"embedding": self._cache[key]}

    def to_dict(self) -> Dict:
        return default_to_dict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "EmbeddingCache":
        return default_from_dict(cls, data)