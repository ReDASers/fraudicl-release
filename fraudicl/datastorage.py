import os
import fraudicl.global_config as global_config
from fraudicl.helper import session_timer
from fraudicl.embedder import EmbeddingManager
from dataclasses import dataclass
from typing import List, Optional
import numpy as np
import h5py
from haystack.dataclasses import Document
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack.components.writers import DocumentWriter
from haystack.components.writers.document_writer import DuplicatePolicy

@dataclass
class VectorDBConfig:
    """Configuration for vector database setup."""
    device_type: str = global_config.General.DEVICE_TYPE
    split_by: str = global_config.VectorDB.SPLIT_METHOD # "word", "sentence", "character"
    split_length: int = global_config.VectorDB.SPLIT_LENGTH
    split_overlap: int = global_config.VectorDB.SPLIT_OVERLAP
    similarity_function: str = global_config.VectorDB.SIMILARITY_FUNCTION

class VectorDBManager:
    """Manages document cleaning, chunking, embedding, and storage in vector database."""

    def __init__(self, embedding_model: Optional[EmbeddingManager] = None, config: Optional[VectorDBConfig] = None) -> None:
        """Initialize with embedding model and config for vector database operations."""
        self.config = config or VectorDBConfig()
        self.embedding_model = embedding_model
        self.document_store = InMemoryDocumentStore(embedding_similarity_function=self.config.similarity_function)
        self.num_documents = None
        self.num_chunks = None

    def setup_vector_database(self, documents: List[Document]) -> InMemoryDocumentStore:
        """Clean, chunk, embed, and store documents; returns populated document store."""
        print("Loading, embedding, and storing processed documents...")
        print("="*50)
        session_timer.start_time(f"setup-vectordb-pipeline")
        processed_docs = self._clean_documents(documents)
        processed_docs = self._chunk_documents(processed_docs)
        processed_docs = self.embedding_model._embed_documents(processed_docs)
        self._store_documents(processed_docs)
        session_timer.stop_time(f"setup-vectordb-pipeline")
        print("="*50)
        print(f"Processed {self._get_lengths(documents, "docu")} documents as {self._get_lengths(processed_docs, 'chunk')} chunks and embedded")
        session_timer.report_time(f"setup-vectordb-pipeline")
        return self.document_store

    def _clean_documents(self, documents: List[Document]) -> List[Document]:
        """Remove any potential prompt template artifacts from documents."""
        print("-"*30)
        print(f"Cleaning {self._get_lengths(documents, "docu")} documents")
        session_timer.start_time(f"cleaning-documents")
        document_cleaner = DocumentCleaner(remove_substrings=global_config.VectorDB.SUBSTRINGS_TO_REMOVE)
        cleaned_result = document_cleaner.run(documents)
        cleaned_docs = cleaned_result["documents"]
        session_timer.stop_time(f"cleaning-documents")
        print(f"Cleaned {self._get_lengths(documents, "docu")} documents")
        session_timer.report_time(f"cleaning-documents")
        print("")
        return cleaned_docs
    
    def _chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Chunks documents into smaller segments for embedding."""
        print("-"*30)
        print(f"Chunking {self._get_lengths(documents, "docu")} documents")
        session_timer.start_time(f"chunking-documents")
        document_splitter = DocumentSplitter(
            split_by=self.config.split_by,
            split_length=self.config.split_length,
            split_overlap=self.config.split_overlap
        )
        split_result = document_splitter.run(documents)
        split_docs = split_result["documents"]
        print(f"Chunked into {self._get_lengths(split_docs, "chunk")} chunks")
        session_timer.stop_time(f"chunking-documents")
        print(f"Chunked {self._get_lengths(documents, "docu")} documents")
        session_timer.report_time(f"chunking-documents")
        print("")
        return split_docs
    
    def _store_documents(self, vectors: List[Document]) -> None:
        """Store vectors in document store."""
        print("-"*30)
        print(f"Storing {self._get_lengths(vectors, "chunk")} vectors")
        session_timer.start_time(f"storing-documents")
        document_writer = DocumentWriter(document_store=self.document_store, policy=DuplicatePolicy.OVERWRITE)
        document_writer.run(vectors)
        session_timer.stop_time(f"storing-documents")
        print(f"Stored {self._get_lengths(vectors, "chunk")} vectors")
        session_timer.report_time(f"storing-documents")
        print("")

    def _get_lengths(self, docu_list: List[Document], mode: str) -> int:
        """Helper method to query and store the number of documents and chunks.

        Always returns the actual length of the passed list, updating the cache.
        """
        length = len(docu_list)
        if mode == "docu":
            self.num_documents = length
        elif mode == "chunk":
            self.num_chunks = length
        return length

    def save_document_store(self, filename: str) -> None:
        """Save document store using HDF5."""
        filepath = f"{global_config.General.LOCAL_SAVED_PATH}/{filename}"
        print("-"*30)
        print(f"Saving document store to {filepath}")
        documents = self.document_store.filter_documents()
        with h5py.File(filepath, "w") as f:
            embeddings = [doc.embedding for doc in documents if doc.embedding is not None]
            if embeddings:
                embeddings_array = np.array(embeddings, dtype=np.float32)
                f.create_dataset("embeddings", data=embeddings_array, compression="gzip", compression_opts=1)
            contents = [doc.content for doc in documents]
            f.create_dataset("contents", data=[c.encode("utf-8") for c in contents])
            domains = [doc.meta.get("domain", "") for doc in documents]
            labels = [doc.meta.get("label", -1) for doc in documents]
            tags = [doc.meta.get("tag", "") for doc in documents]
            f.create_dataset("domains", data=[d.encode("utf-8") for d in domains])
            f.create_dataset("labels", data=labels)
            f.create_dataset("tags", data=[t.encode("utf-8") for t in tags])
            f.attrs["similarity_function"] = self.document_store.embedding_similarity_function
            f.attrs["num_documents"] = len(documents)
            f.attrs["embedding_dim"] = len(embeddings[0]) if embeddings else 0
        print(f"Document store saved with {len(documents)} documents")
        return

    def load_document_store(self, filename: str) -> List[Document]:
        """Load the document store from its HDF5 file under LOCAL_SAVED_PATH.

        Args:
            filename: Name of the document store file.

        Returns:
            List of loaded Document objects.

        Raises:
            FileNotFoundError: if the file does not exist. The store is built from the DiFrauD
                train split (main.py, load_or_build_document_store); there is no hosted copy.
        """
        filepath = f"{global_config.General.LOCAL_SAVED_PATH}/{filename}"
        if not os.path.exists(filepath):
            raise FileNotFoundError(
                f"Document store not found at {filepath}. Run main.py once to build it from the DiFrauD train split."
            )
        print("-"*30)
        print(f"Loading document store from {filepath}")
        with h5py.File(filepath, "r") as f:
            embeddings = f["embeddings"][:] if "embeddings" in f else []
            contents = [c.decode("utf-8") for c in f["contents"][:]]
            domains = [d.decode("utf-8") for d in f["domains"][:]]
            labels = f["labels"][:]
            tags = [t.decode("utf-8") for t in f["tags"][:]]
            similarity_function = f.attrs["similarity_function"]
            num_documents = f.attrs["num_documents"]
        self.document_store = InMemoryDocumentStore(embedding_similarity_function=similarity_function)
        documents = []
        for i, content in enumerate(contents):
            doc = Document(
                content=content,
                embedding=embeddings[i] if i < len(embeddings) else None,
                meta={
                    "domain": domains[i] if i < len(domains) else "",
                    "label": labels[i] if i < len(labels) else -1,
                    "tag": tags[i] if i < len(tags) else ""
                }
            )
            documents.append(doc)
        writer = DocumentWriter(document_store=self.document_store, policy=DuplicatePolicy.OVERWRITE)
        writer.run(documents)
        print(f"Document store loaded with {num_documents} documents")
        return documents
