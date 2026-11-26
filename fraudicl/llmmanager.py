import fraudicl.global_config as global_config
from fraudicl.helper import memclear, session_timer
import json
import os
from dataclasses import dataclass
from typing import Dict, Union, Optional, Any, Tuple, List
import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from haystack.components.generators import HuggingFaceLocalGenerator
from haystack import component, default_to_dict, default_from_dict
import torch.nn.functional as F

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

@dataclass
class LLMConfig:
    """Configuration for LLM loading and inference."""
    device_type: str = global_config.General.DEVICE_TYPE
    max_memory: Union[Dict[str, str], None] = None # set in post_init
    low_cpu_mem_usage: bool = global_config.LLM.LOW_CPU_MEM_USAGE
    use_cache: bool = global_config.LLM.USE_CACHE
    model_key: Union[str, None] = global_config.LLM.DEFAULT_MODEL_KEY
    model_name: Union[str, None] = None
    prompt_profile: Union[str, None] = None
    pad_token_id: int = global_config.LLM.PAD_TOKEN_ID
    trust_remote_code: bool = global_config.LLM.TRUST_REMOTE_CODE
    use_fast_tokenizer: bool = global_config.LLM.USE_FAST_TOKENIZER
    attn_implementation: str = global_config.LLM.ATTN_IMPLEMENTATION
    quantization_mode: Union[str, None] = None
    quant_config_add: Union[BitsAndBytesConfig, None] = None
    max_input_tokens: int = global_config.LLM.MAX_INPUT_TOKENS
    random_seed: int = global_config.General.RANDOM_SEED
    max_new_tokens: int = global_config.LLM.MAX_OUTPUT_TOKENS
    do_sample: bool = global_config.LLM.DO_SAMPLE
    temperature: Union[float, None] = None
    top_p: Union[float, None] = None
    top_k_tokens: Union[int, None] = None
    repetition_penalty: float = global_config.LLM.REPETITION_PENALTY
    num_return_sequences: int = global_config.LLM.NUM_RETURN_SEQUENCES

    # Post init to adjust configs
    def __post_init__(self) -> None:
        """Post-init to set configs."""
        self._apply_model_profile()
        if self.quantization_mode in ("None", "none", "null", ""):
            self.quantization_mode = None
        if self.max_memory is None:
            self.max_memory = global_config.LLM.DEFAULT_MAX_MEMORY.copy()
        self._parse_quantization()
        self._parse_sampling_parameters()

    def _apply_model_profile(self) -> None:
        """Apply model-family defaults from the global LLM registry."""
        profile = {}
        if self.model_name is None:
            profile = global_config.LLM.get_model_profile(self.model_key or global_config.LLM.DEFAULT_MODEL_KEY)
            self.model_name = profile["model_name"]
        elif self.model_name in global_config.LLM.MODEL_ALIASES:
            profile = global_config.LLM.get_model_profile(self.model_name)

        if self.prompt_profile is None:
            self.prompt_profile = profile.get("prompt_profile", "qwen3")

        if profile:
            if self.max_input_tokens == global_config.LLM.MAX_INPUT_TOKENS:
                self.max_input_tokens = profile.get("max_input_tokens", self.max_input_tokens)
            if self.max_new_tokens == global_config.LLM.MAX_OUTPUT_TOKENS:
                self.max_new_tokens = profile.get("max_new_tokens", self.max_new_tokens)
            if self.attn_implementation == global_config.LLM.ATTN_IMPLEMENTATION:
                self.attn_implementation = profile.get("attn_implementation", self.attn_implementation)
            if self.trust_remote_code == global_config.LLM.TRUST_REMOTE_CODE:
                self.trust_remote_code = profile.get("trust_remote_code", self.trust_remote_code)
            if self.use_fast_tokenizer == global_config.LLM.USE_FAST_TOKENIZER:
                self.use_fast_tokenizer = profile.get("use_fast_tokenizer", self.use_fast_tokenizer)
            if self.pad_token_id == global_config.LLM.PAD_TOKEN_ID:
                self.pad_token_id = profile.get("pad_token_id", self.pad_token_id)
            if self.repetition_penalty == global_config.LLM.REPETITION_PENALTY:
                self.repetition_penalty = profile.get("repetition_penalty", self.repetition_penalty)

    def _parse_quantization(self) -> None:
        """Parses quantization parameter."""
        quant_config = None
        if self.quantization_mode is None:
            pass
        elif self.quantization_mode == "4bit":
            quant_config = BitsAndBytesConfig( # extra params to suppress bitandbyte warnings
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        elif self.quantization_mode == "8bit": # is 8bit
            quant_config = BitsAndBytesConfig(
                load_in_8bit=True,
            )
        else:
            raise ValueError(f"Unknown quantization mode: {self.quantization_mode}")
        self.quant_config_add = quant_config

    def _parse_sampling_parameters(self) -> None:
        """Adds sampling parameters if do_sample=True."""
        if self.do_sample:
            if self.temperature is None:
                self.temperature = global_config.LLM.TEMPERATURE
            if self.top_p is None:
                self.top_p = global_config.LLM.TOP_P
            if self.top_k_tokens is None:
                self.top_k_tokens = global_config.LLM.TOP_K_TOKENS

class LLMManager:
    """Manages LLM loading, quantization (4/8-bit), saving, and inference operations."""

    def __init__(self, config: Union[LLMConfig, None] = None) -> None:
        """Initialize with config; model loaded lazily via quantize_LLM() or load_model()."""
        self.config = config or LLMConfig()
        self.model: Union[AutoModelForCausalLM, None] = None
        self.tokenizer: Union[AutoTokenizer, None] = None
        self.passthrough: Union[HuggingFaceLocalGenerator, None] = None
        self.model_info_cache: Union[Dict[str, Any], None] = None

    def _configure_torch_runtime(self) -> None:
        """Enable safe inference runtime optimizations."""
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    def _normalized_quantization_mode(self, quantization_mode: Union[str, None]) -> Union[str, None]:
        """Normalize common string/null spellings of no quantization."""
        if quantization_mode in (None, "None", "none", "null", ""):
            return None
        return quantization_mode

    def _local_model_path(self, quantization_mode: Union[str, None]) -> str:
        """Return a model-specific local save path to avoid collisions across LLMs."""
        quant_tag = quantization_mode or "none"
        model_tag = self.config.model_name.replace("/", "_")
        return f"{global_config.General.LOCAL_SAVED_PATH}/models/{model_tag}_model_{quant_tag}"

    def _legacy_local_model_path(self, quantization_mode: Union[str, None]) -> str:
        """Older flat save path (Saved/model_<quant>), still checked when loading."""
        return f"{global_config.General.LOCAL_SAVED_PATH}/model_{quantization_mode or 'None'}"

    def _legacy_named_model_path(self, quantization_mode: Union[str, None]) -> str:
        """Older per-model save path (Saved/<model>_model_<quant>), still checked when loading."""
        quant_tag = quantization_mode or "none"
        model_tag = self.config.model_name.replace("/", "_")
        return f"{global_config.General.LOCAL_SAVED_PATH}/{model_tag}_model_{quant_tag}"

    def _config_matches_requested_model(self, config_filepath: str) -> bool:
        """Return True only when a saved config belongs to the requested model."""
        try:
            with open(config_filepath, "r") as f:
                saved_config = json.load(f)
        except (OSError, json.JSONDecodeError):
            return False
        return saved_config.get("model_name") == self.config.model_name

    def quantize_LLM(self, save_mode: bool = True) -> None:
        """Load and quantize the LLM from HuggingFace Hub (for first-time setup)."""
        print("Loading and quantizing LLM")
        print("=" * 50)
        session_timer.start_time(f"loading-all-llm")
        self.tokenizer = self._setup_tokenizer()
        self.model = self._setup_model(save_mode)
        session_timer.stop_time(f"loading-all-llm")
        print("=" * 50)
        print("LLM loading and quantizing completed")
        session_timer.report_time(f"loading-all-llm")

    def _setup_tokenizer(self) -> AutoTokenizer:
        """Load and configure the tokenizer."""
        print("-" * 30)
        print(f"Loading tokenizer")
        session_timer.start_time(f"loading-llm-tokenizer")
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=self.config.trust_remote_code,
            use_fast=self.config.use_fast_tokenizer,
            token=HF_TOKEN,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token_id = self.config.pad_token_id
        self.tokenizer = tokenizer
        session_timer.stop_time(f"loading-llm-tokenizer")
        print(f"Tokenizer loaded with vocab size: {tokenizer.vocab_size}")
        session_timer.report_time(f"loading-llm-tokenizer")
        print("")
        return tokenizer

    def _setup_model(self, save_mode: bool) -> AutoModelForCausalLM:
        """Load and configure the model."""
        print("-" * 30)
        print(f"Loading & quantizing model {self.config.model_name} in {self.config.quantization_mode}")
        session_timer.start_time(f"loading-llm-model")
        if save_mode: # safe load settings if preparing for save
            self.config.device_type = "cpu"
            self.config.low_cpu_mem_usage = False
        model_kwargs = {
            "device_map": self.config.device_type,
            "max_memory": self.config.max_memory,
            "pad_token_id": self.config.pad_token_id,
            "low_cpu_mem_usage": self.config.low_cpu_mem_usage,
            "trust_remote_code": self.config.trust_remote_code
        }
        if self.config.attn_implementation is not None:
            model_kwargs["attn_implementation"] = self.config.attn_implementation
        if self.config.quant_config_add is not None:
            model_kwargs["quantization_config"] = self.config.quant_config_add
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            token=HF_TOKEN,
            **model_kwargs
        )
        model.eval()
        self.model = model
        session_timer.stop_time(f"loading-llm-model")
        self._report_model_info()
        session_timer.report_time(f"loading-llm-model")
        print("")
        return model

    def _report_model_info(self) -> None:
        """Report model info."""
        if self.model_info_cache is None:
            self._calculate_model_info()
        print(f"Total parameters: {self.model_info_cache['total_params']:,}")
        print(f"Trainable parameters: {self.model_info_cache['trainable_params']:,}")
        print(f"Parameter dtype(s): {', '.join(self.model_info_cache['dtype_variants'])}")
        print(f"Active dtype: {self.model_info_cache['dtype']}")
        print(f"Quantization mode: {self.model_info_cache['quantization_mode']}")
        print(f"Est. memory usage: {self.model_info_cache['estimated_memory_gb']:.2f} GB")
        
        if "gpu_memory_allocated_gb" in self.model_info_cache:
            print(f"GPU memory reserved: {self.model_info_cache['gpu_memory_reserved_gb']:.2f} GB")
            print(f"GPU memory allocated: {self.model_info_cache['gpu_memory_allocated_gb']:.2f} GB")

        return

    def _calculate_model_info(self) -> None:
        """Calculate model information and cache it."""
        print("~" * 15)
        print(f"Caching model info")
        session_timer.start_time(f"getting-model-info")
        total_params = 0
        trainable_params = 0
        total_bytes = 0
        dtype_set = set()
        dtype_size_map = {
            torch.float32: 4, torch.float16: 2, torch.bfloat16: 2,
            torch.int8: 1, torch.uint8: 1,
            torch.qint8: 1, torch.quint8: 1
        }
        for param in self.model.parameters():
            param_count = param.numel()
            total_params += param_count
            if param.requires_grad:
                trainable_params += param_count
            dtype_set.add(param.dtype)
            param_bytes = dtype_size_map.get(param.dtype, 4)
            total_bytes += param_count * param_bytes
        try:
            estimated_model_size_gb = self.model.get_memory_footprint() / (1024**3)
        except (AttributeError, RuntimeError):
            estimated_model_size_gb = total_bytes / (1024**3)
        first_param = next(self.model.parameters())
        dtype_str = str(first_param.dtype)
        self.model_info_cache = {
            "total_params": total_params,
            "trainable_params": trainable_params,
            "dtype": dtype_str,
            "estimated_memory_gb": estimated_model_size_gb,
            "quantization_mode": self.config.quantization_mode or "none",
            "dtype_variants": [str(dt) for dt in dtype_set]
        }
        if torch.cuda.is_available():
            self.model_info_cache.update({
                "gpu_memory_allocated_gb": torch.cuda.memory_allocated() / 1024**3,
                "gpu_memory_reserved_gb": torch.cuda.memory_reserved() / 1024**3,
            })
        session_timer.stop_time(f"getting-model-info")
        print("Model info cached")
        session_timer.report_time(f"getting-model-info")
        print("~" * 15)

    def save_model(self, use_safetensors: bool = True) -> None:
        """Save model using HuggingFace format (safetensors)."""
        filepath = self._local_model_path(self.config.quantization_mode)
        print("-" * 30)
        print(f"Saving model to {filepath} (safetensors: {use_safetensors})")
        session_timer.start_time(f"save-quantized-llm")
        self.model.save_pretrained(
            filepath,
            safe_serialization=use_safetensors,
            max_shard_size="5GB"
        )
        self.tokenizer.save_pretrained(filepath)
        config_data = {
            "model_key": self.config.model_key,
            "model_name": self.config.model_name,
            "prompt_profile": self.config.prompt_profile,
            "quantization_config": {
                "use_quantization": self._parse_quantization_parameter(),
                "quantization_mode": self.config.quantization_mode
            },
            "device_config": {
                "device_type": self.config.device_type,
                "max_memory": self.config.max_memory,
                "low_cpu_mem_usage": self.config.low_cpu_mem_usage
            },
            "model_params": {
                "attn_implementation": self.config.attn_implementation,
                "pad_token_id": self.config.pad_token_id,
                "trust_remote_code": self.config.trust_remote_code
            },
            "model_info": self.model_info_cache
        }

        config_filepath = f"{filepath}/model_{self.config.quantization_mode or 'none'}_config.json"
        with open(config_filepath, "w") as f:
            json.dump(config_data, f, indent=2, default=str)
        session_timer.stop_time(f"save-quantized-llm")
        print(f"Model saved successfully to {filepath}")
        session_timer.report_time(f"save-quantized-llm")
        return

    def _parse_quantization_parameter(self) -> bool:
        """Helper function for setting use_quantization parameter."""
        return self.config.quantization_mode is not None

    def load_model(self, quantization_mode: str = "None", from_hub: bool = False) -> None:
        """
        Load model using optimized HuggingFace format.

        Args:
            quantization_mode: "4bit", "8bit", or "None"
            from_hub: If True, load directly from HF Hub and quantize on-the-fly.
                      If False, load from local ./Saved/ directory.
        """
        print("-" * 30)
        quantization_mode = self._normalized_quantization_mode(quantization_mode)
        self._configure_torch_runtime()

        if from_hub:
            model_name = self.config.model_name
            print(f"Loading model from HF Hub: {model_name}")
            session_timer.start_time(f"load-quantized-llm")
            self.config.quantization_mode = quantization_mode
            self.config._parse_quantization()
            if torch.cuda.is_available():
                self.config.device_type = "cuda:0"
            else:
                self.config.device_type = "auto"
            print("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=self.config.trust_remote_code,
                use_fast=self.config.use_fast_tokenizer,
                token=HF_TOKEN,
            )
        else:
            filepath = self._local_model_path(quantization_mode)
            config_filepath = f"{filepath}/model_{quantization_mode or 'none'}_config.json"
            if not self._config_matches_requested_model(config_filepath):
                legacy_named_filepath = self._legacy_named_model_path(quantization_mode)
                legacy_named_config = f"{legacy_named_filepath}/model_{quantization_mode or 'none'}_config.json"
                if self._config_matches_requested_model(legacy_named_config):
                    filepath = legacy_named_filepath
                    config_filepath = legacy_named_config
                else:
                    filepath = self._legacy_local_model_path(quantization_mode)
                    config_filepath = f"{filepath}/model_{quantization_mode or 'None'}_config.json"
                    if not self._config_matches_requested_model(config_filepath):
                        raise FileNotFoundError(
                            f"No local checkpoint for {self.config.model_name} "
                            f"with quantization {quantization_mode or 'none'} under "
                            f"{global_config.General.LOCAL_SAVED_PATH}"
                        )
            print(f"Loading model from {filepath}")
            session_timer.start_time(f"load-quantized-llm")
            with open(config_filepath, "r") as f:
                saved_config = json.load(f)
            print(f"Original model: {saved_config['model_name']}")
            self.config.model_key = saved_config.get("model_key", self.config.model_key)
            self.config.model_name = saved_config.get("model_name", self.config.model_name)
            self.config.prompt_profile = saved_config.get("prompt_profile", self.config.prompt_profile)
            if torch.cuda.is_available():
                self.config.device_type = "cuda:0"
            else:
                self.config.device_type = "auto"
            quant_mode = saved_config.get("quantization_config", {}).get("quantization_mode", None)
            self.config.quantization_mode = self._normalized_quantization_mode(quant_mode)
            self.model_info_cache = saved_config.get("model_info", {})
            print("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                filepath,
                trust_remote_code=self.config.trust_remote_code,
                use_fast=self.config.use_fast_tokenizer
            )
            model_name = filepath

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = self.config.pad_token_id
        model_kwargs = {
            "device_map": self.config.device_type,
            "max_memory": self.config.max_memory,
            "low_cpu_mem_usage": self.config.low_cpu_mem_usage,
            "trust_remote_code": self.config.trust_remote_code
        }
        if self.config.quant_config_add is None and torch.cuda.is_available():
            model_kwargs["dtype"] = torch.float16
        if self.config.attn_implementation:
            model_kwargs["attn_implementation"] = self.config.attn_implementation
        if from_hub and self.config.quant_config_add is not None:
            model_kwargs["quantization_config"] = self.config.quant_config_add
        print("Loading model...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            token=HF_TOKEN,
            **model_kwargs
        )
        device = next(self.model.parameters()).device
        print("Optimizing for inference...")
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.config.use_cache = True
        session_timer.stop_time(f"load-quantized-llm")
        print(f"Model loaded on device: {device}")
        if torch.cuda.is_available() and device.type == "cuda":
            print(f"CUDA device: {torch.cuda.get_device_name(device)}")
            print(f"CUDA memory allocated: {torch.cuda.memory_allocated(device) / 1024**3:.2f} GB")
            print(f"CUDA memory reserved: {torch.cuda.memory_reserved(device) / 1024**3:.2f} GB")
        print("Model loaded and optimized successfully")
        session_timer.report_time(f"load-quantized-llm")
        return

    def _prepare_generation_kwargs(self) -> Dict[str, Any]:
        """Prepare generation keyword arguments for LLM."""
        pad_token_id = (
            self.tokenizer.pad_token_id
            if self.tokenizer is not None and self.tokenizer.pad_token_id is not None
            else self.config.pad_token_id
        )
        generation_kwargs = {
            "max_new_tokens": self.config.max_new_tokens,
            "do_sample": self.config.do_sample,
            "num_return_sequences": self.config.num_return_sequences,
            "use_cache": self.config.use_cache,
            "repetition_penalty": self.config.repetition_penalty,
            "pad_token_id": pad_token_id
        }
        
        # Add sampling parameters if sampling is enabled
        if self.config.do_sample:
            generation_kwargs.update({
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "top_k": self.config.top_k_tokens
            })
        
        return generation_kwargs
    
    def _predict_label_probs(self, prompt: str) -> Dict[str, float]:
        """Returns {canonical_label: probability} for next-token label variants. Used for demo_order.

        This preserves the original first-token DEmO approximation while allowing
        prompt-compatible surface variants to map to canonical dataset labels.
        """
        label_variants = {
            "non-deceptive": ["non-deceptive", "not deceptive"],
            "deceptive": ["deceptive"],
        }
        device = next(self.model.parameters()).device
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_input_tokens
        ).to(device)

        with torch.inference_mode():
            outputs = self.model(**inputs)
            next_token_logits = outputs.logits[0, -1, :]

        label_logprobs = torch.stack([
            self._first_token_variant_logit(next_token_logits, label, label_variants)
            for label in list(global_config.Database.LABEL_MAPPING.values())[:2]
        ])
        label_probs = F.softmax(label_logprobs, dim=0).float().cpu().numpy()
        return {
            label: float(prob)
            for label, prob in zip(list(global_config.Database.LABEL_MAPPING.values())[:2], label_probs)
        }

    def _first_token_variant_logit(
        self,
        next_token_logits: torch.Tensor,
        canonical_label: str,
        label_variants: Dict[str, List[str]],
    ) -> torch.Tensor:
        """Return the strongest next-token logit across surface variants of a label."""
        token_logits = []
        for variant in label_variants.get(canonical_label, [canonical_label]):
            for candidate in (variant, f" {variant}"):
                token_ids = self.tokenizer(candidate, add_special_tokens=False)["input_ids"]
                if token_ids:
                    token_logits.append(next_token_logits[token_ids[0]])
        if not token_logits:
            raise ValueError(f"Label '{canonical_label}' tokenizes to empty sequence")
        return torch.stack(token_logits).max()
