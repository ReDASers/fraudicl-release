import fraudicl.global_config as global_config
from fraudicl.helper import memclear, session_timer
from fraudicl.embedder import EmbeddingManager, EmbeddingCache
from fraudicl.datastorage import VectorDBManager
from fraudicl.retrievalmanager import RetrievalManager
from fraudicl.ordermanager import OrderManager
from fraudicl.promptmanager import PromptManager
from fraudicl.llmmanager import LLMConfig, LLMManager
from fraudicl.evaluationmanager import AnswerParser
from typing import Optional, Dict, Any, Tuple, List
from dataclasses import dataclass
from transformers import logging
from haystack import Pipeline, component
import torch

@dataclass
class PipelineConfig:
    """Configuration for loading RAG components."""
    embedding_model: EmbeddingManager = None
    retrieval_model: RetrievalManager = None
    order_model: OrderManager = None
    prompt_template: PromptManager = None
    llm_model: LLMManager = None


@component
class DirectLLMGenerator:
    """Haystack component that generates directly with a pre-loaded HF causal LM."""

    def __init__(self, llm_manager: LLMManager, generation_kwargs: Dict[str, Any]) -> None:
        self.llm_manager = llm_manager
        self.generation_kwargs = {
            key: value
            for key, value in generation_kwargs.items()
            if key not in {"return_full_text"}
        }
        # When set, run() returns the prompt without generating. Batched generation
        # needs the assembled prompt only: Haystack runs every downstream component
        # regardless of include_outputs_from, and components cannot be shared across
        # pipelines, so the generator is skipped in place rather than detached.
        self.prepare_only = False

    @component.output_types(replies=List[str], raw_prompt=str)
    def run(self, prompt: str) -> Dict[str, Any]:
        if self.prepare_only:
            return {"replies": [""], "raw_prompt": prompt}
        tokenizer = self.llm_manager.tokenizer
        model = self.llm_manager.model
        device = next(model.parameters()).device
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.llm_manager.config.max_input_tokens
        ).to(device)

        with torch.inference_mode():
            outputs = model.generate(**inputs, **self.generation_kwargs)

        prompt_length = inputs["input_ids"].shape[1]
        generated_tokens = outputs[:, prompt_length:]
        replies = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        return {
            "replies": [reply.strip() for reply in replies],
            "raw_prompt": prompt,
        }

    def run_batch(self, prompts: List[str]) -> List[Dict[str, Any]]:
        """Generate for several prompts in one forward pass.

        Batch-1 decoding of a 7-9B model is memory-bandwidth bound: the weights are
        read in full to advance a single sequence by one token, which is why the GPU
        sits near 13% utilisation. Batching amortises that read across sequences.

        Decoder-only models must be LEFT-padded for batched generation: generation
        continues from the last position of each row, so right-padding would continue
        from pad tokens and silently corrupt short prompts. The attention mask keeps
        the pads from being attended to, and slicing the continuation at the (now
        uniform) padded input width yields exactly the single-prompt result.
        """
        if not prompts:
            return []

        tokenizer = self.llm_manager.tokenizer
        model = self.llm_manager.model
        device = next(model.parameters()).device

        previous_side = tokenizer.padding_side
        tokenizer.padding_side = "left"
        try:
            inputs = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.llm_manager.config.max_input_tokens,
            ).to(device)

            with torch.inference_mode():
                outputs = model.generate(**inputs, **self.generation_kwargs)
        finally:
            tokenizer.padding_side = previous_side

        # Left padding makes every row's prompt end at the same index, so one offset
        # slices the generated continuation for the whole batch.
        prompt_length = inputs["input_ids"].shape[1]
        generated_tokens = outputs[:, prompt_length:]
        replies = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        return [
            {"replies": [reply.strip()], "raw_prompt": prompt}
            for reply, prompt in zip(replies, prompts)
        ]

class PipelineManager:
    """Manages the complete RAG pipeline for deception detection."""
    
    def __init__(self, config: Optional[PipelineConfig] = None) -> None:
        """
        Initialize the RAG manager with existing component managers.
        
        Instance attributes:
            config              config for RAG pipeline
            pipeline            contains the finalized pipeline
        """
        self.config = config or PipelineConfig()
        self.pipeline = None
        # Same components as `pipeline` but stopping at the prompt; used by batched
        # generation so fetching a prompt does not also run the generator.
        self.prepare_pipeline = None
        self.generator = None
        self.embedding_cache: Optional[EmbeddingCache] = None

    def setup_pipeline(self) -> None:
        """Set up the complete RAG pipeline using existing component managers."""
        print(f"Creating full pipeline...")
        print("="*50)

        # Clear memory before setup & suppress temp warnings
        memclear(False)
        logging.set_verbosity_error()
        session_timer.start_time(f"creating-full-pipeline")
        rag = Pipeline()
        # Add cache-backed text embedder (populated via pre_embed before pipeline runs)
        self.embedding_cache = EmbeddingCache()
        print("Adding text embedder (cache-backed; call pre_embed before running)")
        rag.add_component("fraudicl_text_embedder", self.embedding_cache)
        if self.config.retrieval_model is None:
            session_db = VectorDBManager()
            session_documents = session_db.load_document_store("document_store")
            session_retriever = RetrievalManager(session_documents)
            del session_db
        else:
            session_retriever = self.config.retrieval_model

        print(f"Adding document retriever")
        rag.add_component(
            "fraudicl_retriever",
            session_retriever # InMemoryDocumentStore
        )
        del session_retriever

        if self.config.llm_model is None:
            quantization_mode = "4bit"
            llm_config = LLMConfig(quantization_mode=quantization_mode)
            session_llm = LLMManager(config=llm_config)
            session_llm.load_model(quantization_mode=quantization_mode)
        else:
            session_llm = self.config.llm_model

        if self.config.prompt_template is None:
            session_prompter = PromptManager(prompt_profile=session_llm.config.prompt_profile)
        else:
            session_prompter = self.config.prompt_template

        print(f"Adding prompt selector")
        rag.add_component(
            "fraudicl_prompt_selector",
            session_prompter # PromptManager
        )

        if self.config.order_model is None:
            session_orderer = OrderManager(
                prompt_manager=session_prompter,
                llm_manager=session_llm,
            )
        else:
            session_orderer = self.config.order_model

        print(f"Adding order model")
        rag.add_component(
            "fraudicl_orderer",
            session_orderer # OrderManager
        )
        del session_orderer

        generation_kwargs = session_llm._prepare_generation_kwargs()
        generator = DirectLLMGenerator(session_llm, generation_kwargs)
        print(f"Adding LLM generator")
        rag.add_component(
            "fraudicl_llm",
            generator # DirectLLMGenerator
        )
        rag.connect("fraudicl_text_embedder.embedding", "fraudicl_retriever.user_query")
        rag.connect("fraudicl_retriever.documents", "fraudicl_orderer.documents")
        rag.connect("fraudicl_orderer.documents", "fraudicl_prompt_selector.top_k_examples")
        rag.connect("fraudicl_prompt_selector.prompt", "fraudicl_llm.prompt")
        self.pipeline = rag
        self.generator = generator
        memclear(False)
        session_timer.stop_time(f"creating-full-pipeline")
        print("="*50)
        print("Pipeline complete")
        session_timer.report_time(f"creating-full-pipeline")

    def pre_embed(self, texts: List[str]) -> None:
        """Batch-embed all query texts once and populate the EmbeddingCache before the pipeline loop."""
        if self.embedding_cache is None:
            raise RuntimeError("Call setup_pipeline() before pre_embed().")
        # Strip to match the pipeline lookup (text.strip() in _analyze_text), so the cached
        # embedding is of exactly the text used downstream and lookups can't miss on whitespace.
        unique_texts = list(dict.fromkeys(text.strip() for text in texts))  # dedup, preserve order
        embedding_manager = self.config.embedding_model or EmbeddingManager()
        cache = embedding_manager.batch_embed_texts(unique_texts)
        self.embedding_cache.populate(cache)
        print(f"Embedding cache populated with {len(cache)} entries.")
