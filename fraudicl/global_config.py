class LettuceDetect:
    """Configuration for LettuceDetect hallucination grounding checker."""
    DEFAULT_MODEL = "KRLabsOrg/lettucedect-base-modernbert-en-v1"
    LARGE_MODEL = "KRLabsOrg/lettucedect-large-modernbert-en-v1"

    # Which pieces count as supporting *evidence* when scoring a rationale.
    #
    # LettuceDetect takes three inputs: context, question, and answer. The text
    # under evaluation is passed as `question` in every mode. `question` steers span detection
    # without counting as support, so only what lands in `context` can ground a
    # rationale. These modes vary `context` alone; `question` and `answer` are
    # identical across all of them.
    #
    #   "demos"            -- retrieved demonstrations only. Reproduces the
    #                         paper's numbers ("demonstration-relative
    #                         departure"). A rationale reasoning correctly from
    #                         the evaluated text still scores as departing.
    #   "demos+query"      -- adds the evaluated text. It is the same kind of
    #                         object as the demonstrations (a document with
    #                         content), so observations about it become
    #                         checkable rather than automatically unsupported.
    #   "demos+query+def"  -- also adds the deception definition. The system
    #                         prompt tells the model to "explicitly use both the
    #                         deception definition and the example context", so
    #                         this is the only mode whose evidence set matches
    #                         what the model was licensed to reason from.
    #                         No-ops for definition_type -1/0, which carry no
    #                         definition text (those rows fall back to
    #                         "demos+query" rather than being scored twice).
    #
    # Absolute levels are NOT comparable across modes -- adding evidence changes
    # what counts as supported by construction. Compare within a mode.
    CONTEXT_MODES = ("demos", "demos+query", "demos+query+def")

    # Modes scored on every run unless --grounding-modes overrides. All three, so
    # every mode is available without re-running generation.
    DEFAULT_SCORED_MODES = ("demos", "demos+query", "demos+query+def")

    # The mode that owns the unsuffixed columns (`grounded`, `hallucination_score`,
    # `hallucinated_fraction`, `hallucinated_spans`). "demos" keeps those columns
    # meaning what they mean in the paper's tables; other modes are appended as
    # suffixed column blocks.
    PRIMARY_CONTEXT_MODE = "demos"

    # Back-compat for callers that pass no context_mode (e.g. a bare
    # HallucinationChecker()). Matches PRIMARY_CONTEXT_MODE so a default-constructed
    # checker reproduces the published scoring.
    DEFAULT_CONTEXT_MODE = "demos"

    # CLI/column token per mode. Used for --grounding-modes values and to build
    # suffixed column names (e.g. "hallucinated_fraction_demos_query").
    MODE_SLUGS = {
        "demos": "demos",
        "demos+query": "demos_query",
        "demos+query+def": "demos_query_def",
    }
    SLUG_TO_MODE = {v: k for k, v in MODE_SLUGS.items()}

    # Per-mode grounding fields, in the order HallucinationChecker.check returns them.
    GROUNDING_FIELDS = (
        "grounded",
        "hallucination_score",
        "hallucinated_fraction",
        "hallucinated_spans",
    )

    # Prefixes that mark each context segment's provenance for the detector.
    QUERY_PREFIX = "Text under evaluation: "
    DEFINITION_PREFIX = "Deception definition: "

    @classmethod
    def resolve_modes(cls, tokens):
        """Map CLI tokens (slugs or raw mode strings) to canonical mode names.

        Accepts either form so `--grounding-modes demos_query` and
        `--grounding-modes demos+query` both work. Order is preserved and
        duplicates are dropped, since column order follows the requested order.
        """
        if not tokens:
            return list(cls.DEFAULT_SCORED_MODES)
        resolved = []
        for tok in tokens:
            mode = cls.SLUG_TO_MODE.get(tok, tok)
            if mode not in cls.CONTEXT_MODES:
                raise ValueError(
                    f"Unknown grounding mode {tok!r}. Valid: "
                    f"{sorted(cls.MODE_SLUGS.values())}"
                )
            if mode not in resolved:
                resolved.append(mode)
        return resolved

    @classmethod
    def column_name(cls, field, mode):
        """Column name for one grounding field under one mode.

        The primary mode owns the unsuffixed names so existing consumers keep
        working; every other mode gets a `_<slug>` suffix.
        """
        if mode == cls.PRIMARY_CONTEXT_MODE:
            return field
        return f"{field}_{cls.MODE_SLUGS[mode]}"

    @classmethod
    def columns_for(cls, modes):
        """All grounding column names for `modes`, grouped mode by mode."""
        return [cls.column_name(f, m) for m in modes for f in cls.GROUNDING_FIELDS]

class Runtime:
    """Runtime constants."""
    # Numerical stability
    EPSILON = 1e-8              # Prevents division by zero in cosine similarity
    LOG_EPSILON = 1e-10         # Prevents log(0) in entropy calculations

    # Memory management
    MEMORY_CLEANUP_INTERVAL = 50    # Iterations between GPU memory cleanup (balances overhead vs fragmentation)

    # Data export limits
    CSV_CELL_CHAR_LIMIT = 32000     # Excel/CSV cell limit (~32,767 chars) to prevent truncation

    # Permutation sampling
    PERM_SAMPLE_MULTIPLIER = 10     # Max attempts = samples * multiplier (birthday paradox bound)

class General:
    """Miscellanous global constants."""
    # Filepaths
    LOCAL_DIFRAUD_PATH = "redasers/difraud"
    LOCAL_OUTPUTS_PATH = "./Outputs"
    LOCAL_PROCESSING_PATH = "./Outputs/Processing"  # fixed; never overridden by experiment-specific output paths
    LOCAL_SAVED_PATH = "./Saved"
    LOCAL_WALKTHROUGH_IMAGES = "./Images/Walkthrough"

    # Random seed
    RANDOM_SEED = 99

    # Device config
    DEVICE_TYPE = "cuda:0" # auto, cpu, cuda:0

class Database:
    """Global constants for pre-processing database."""
    # Keeping only datapoints with n words, where MIN_WORDS <= n <= MAX_WORDS 
    MAX_WORDS = 900
    MIN_WORDS = 5

    # Domains and domain tags lookup
    DOMAIN_MAPPING = {
        "fake_news": "news", 
        "phishing": "email", 
        "sms": "sms", 
        "twitter_rumours": "twitter", 
        "job_scams": "message boards", 
        "product_reviews": "product reviews", 
        "political_statements": "political statements"
    }

    # Label and label value lookup
    LABEL_MAPPING = {
        0: "non-deceptive",
        1: "deceptive",
        -1: "unknown"
    }

class Embedding:
    """Global constants for setting up embedder."""
    # Embedding vectors
    MODEL = "avsolatorio/GIST-small-Embedding-v0"
    DIMENSIONS = 384
    NORMALIZE_EMBEDDINGS = False
    BATCH_SIZE = 512

class VectorDB:
    """Global constants for setting up a vector database."""
    # Document chunking
    SPLIT_METHOD = "word"
    SPLIT_LENGTH = 200
    SPLIT_OVERLAP = 10

    # Retrieval
    SIMILARITY_FUNCTION = "cosine"

    # Specific prompt artifacts to remove from data
    SUBSTRINGS_TO_REMOVE = [
        "Label:",
        "Reasoning:",
        "Context:",
        "### User:",
        "### System:",
        "### Assistant:",
        "<|im_start|>user",
        "<|im_start|>system",
        "<|im_start|>assistant",
        "<|im_end|>",
        "/no_think",
        "Text:",
        "Query:",
        "Question:",
        "Answer:", 
        "</s>",
        "Instruction:"
    ]

class Retrieval:
    """Global variables for retrieval."""
    # Retrieval parameters
    TOP_K = 4

class Order:
    """Global constants for ordering."""
    # DEmO parameters
    PI_SAMPLE = 24 # num of permutations to sample
    TOP_K_ENTROPY = 4 # num of top candidates by entropy score/label fairness to move to next step
    CONTENT_FREE_TOKENS = ["[MASK]", "[N/A]", ""] # tokens to consider as content-free for entropy calculation

class BasePrompt:
    """Shared prompt constants for deception-detection prompt templates."""

    # Deception definitions
    DEFINITION_1: str = "Any textual content that aims to misconstrue an affair in a deliberate way that causes the reader to be at a disadvantage directly or indirectly."
    DEFINITION_2: str = "An intentional act of manipulation to gain compliance. Thus, it has at least one source, one target, and one goal. The source intentionally manipulates the target into beliefs, actions, or both intended to achieve the goal(s)."
    DEFINITION_3: str = "Deception is an act of manipulation designed to gain compliance such that exposing the manipulation or the goal(s) of compliance significantly decreases the chance of compliance. It has at least one source, one target, and one goal. The source manipulates the target into beliefs, actions, or both intended to achieve the goals."

class QwenPrompt(BasePrompt):
    """Prompt templates using Qwen chat-control tokens."""

    # Prompt Texts (FRAUD-ICL)
    SYSTEM_MSG = """<|im_start|>system
You are an AI Assistant with expertise in the detection of digital deception.

Deception is defined as follows:
{{definition}}

Your approach to deception detection:
Using the deception definition and the provided ground truth examples, evaluate the query text and categorize it as either 'deceptive', 'not deceptive', or 'unknown' (if it cannot be determined). Explicitly use both the deception definition and the example context in making your decision; do not use any internal or external knowledge that you cannot verify or cite.

Ground Truth Examples:
{% for doc in documents %}
<|im_end|>
<|im_start|>user
/no_think
Instruction: Using the deception definition and the provided ground truth examples, label the provided text as: 'deceptive', 'not deceptive', or 'unknown'.
{{doc.content}}
<|im_end|>
<|im_start|>assistant
Label: {{doc.meta['tag']}}
{% endfor %}"""

    USER_INPUT = """<|im_start|>user
/no_think
Instruction: Using the deception definition and the provided ground truth examples, label the provided text as: 'deceptive', 'not deceptive', or 'unknown'. On a new line, give a step-by-step breakdown of your reasoning for the decision, citing the deception definition and the ground truth examples where appropriate.
{{text}}
<|im_end|>
<|im_start|>assistant
Label:"""

    SYSTEM_MSG_DEFINITIONLESS = """<|im_start|>system
You are an AI Assistant with expertise in the detection of digital deception. 

Your approach to deception detection:
Using the provided ground truth examples, evaluate the query text and categorize it as either 'deceptive', 'not deceptive', or 'unknown' (if it cannot be determined). Explicitly use the example context in making your decision; do not use any internal or external knowledge that you cannot verify or cite.

Ground Truth Examples:
{% for doc in documents %}
<|im_end|>
<|im_start|>user
/no_think
Instruction: Using the provided ground truth examples, label the provided text as: 'deceptive', 'not deceptive', or 'unknown'.
{{doc.content}}
<|im_end|>
<|im_start|>assistant
Label: {{doc.meta['tag']}}
{% endfor %}"""

    USER_INPUT_DEFINITIONLESS = """<|im_start|>user
/no_think
Instruction: Using the provided ground truth examples, label the provided text as: 'deceptive', 'not deceptive', or 'unknown'. On a new line, give a step-by-step breakdown of your reasoning for the decision, citing the ground truth examples where appropriate.
{{text}}
<|im_end|>
<|im_start|>assistant
Label:"""

    SYSTEM_MSG_ZEROSHOT = """<|im_start|>system
You are an AI Assistant with expertise in the detection of digital deception. 

Your approach to deception detection:
Evaluate the query text and categorize it as either 'deceptive', 'not deceptive', or 'unknown' (if it cannot be determined). Do not use any internal or external knowledge you cannot verify and cite.<|im_end|>
"""

    USER_INPUT_ZEROSHOT = """<|im_start|>user
/no_think
Instruction: Label the provided text as: 'deceptive', 'not deceptive', or 'unknown'. On a new line, give a step-by-step breakdown of your reasoning.
{{text}}
<|im_end|>
<|im_start|>assistant
Label:"""

    SYSTEM_MSG_OLD = """<|im_start|>system
You are an AI Assistant with expertise in the detection of digital deception. You use the following definition of deception: {{definition}}
    
Deceptive texts aim to achieve malicious objectives and include dishonest content such as misinformation, fake news, propaganda, spam, scams, phishing attempts, opinion spam, misinformation and false rumours on twitter and other social media, opinion spam or spam product reviews
    
Deception in text is defined as follows:
{{definition}}

Approach to Detection:
Using only the definition of deception ahd the context of ground truth texts, evaluate the text and categorize it as either deceptive or not deceptive. If the answer cannot be determined, say unknown. DO use definition of deception and context, do NOT use any internal or external knowledge you cannot verify and cite. Give a step-by-step explanation with references to specific ground truth examples and the definition of deception.
Use the following context and the deinition of deception within it to help respond to the instruction from the user.
IF you are not sure of the answer, say I do not know.

Use the following context and the definition of deception to help respond to the instruction from the user:
Context:
{% for doc in documents %}
<|im_end|>
<|im_start|>user
/no_think
Instruction: Relying only on the definition of deception and context, label the text below as: deceptive, not deceptive.
{{doc.content}}
<|im_end|>
<|im_start|>assistant
Label: {{doc.meta['tag']}}
{% endfor %}"""

    USER_INPUT_OLD = """<|im_start|>user
/no_think
Instruction: Relying only on the definition of deception and context, label the text below as: deceptive, not deceptive. If you are unsurse of the answer, say unknown. On a new line, explain your reasoning step-by-step, citing the definition of deception where appropriate.
{{text}}
<|im_end|>
<|im_start|>assistant
Label:"""


class NeuralChatPrompt(BasePrompt):
    """Prompt templates using Intel NeuralChat/Mistral-style instruction tags."""

    SYSTEM_MSG ="""You are an AI Assistant with expertise in the detection of digital deception. 

Deception is defined as follows:
{{definition}}

Your approach to deception detection:
Using the deception definition and the provided ground truth examples, evaluate the query text and categorize it as either 'deceptive', 'not deceptive', or 'unknown' (if it cannot be determined). Explicitly use both the deception definition and the example context in making your decision; do not use any internal or external knowledge that you cannot verify or cite.

Ground Truth Examples:
{% for doc in documents %}
### User:
Instruction: Using the deception definition and the provided ground truth examples, label the provided text as: 'deceptive', 'not deceptive', or 'unknown'.
{{doc.content}}

### Assistant:
Label: {{doc.meta['tag']}}
{% endfor %}"""

    USER_INPUT = """### User:
Instruction: Using the deception definition and the provided ground truth examples, label the provided text as: 'deceptive', 'not deceptive', or 'unknown'. On a new line, give a step-by-step breakdown of your reasoning for the decision, citing the deception definition and the ground truth examples where appropriate.
{{text}}

### Assistant:
Label:"""

    SYSTEM_MSG_DEFINITIONLESS = """You are an AI Assistant with expertise in the detection of digital deception. 

Your approach to deception detection:
Using the provided ground truth examples, evaluate the query text and categorize it as either 'deceptive', 'not deceptive', or 'unknown' (if it cannot be determined). Explicitly use the example context in making your decision; do not use any internal or external knowledge that you cannot verify or cite.

Ground Truth Examples:
{% for doc in documents %}
### User:
Instruction: Using the provided ground truth examples, label the provided text as: 'deceptive', 'not deceptive', or 'unknown'.
{{doc.content}}

### Assistant:
Label: {{doc.meta['tag']}}
{% endfor %}"""

    USER_INPUT_DEFINITIONLESS = """### User:
Instruction: Using the provided ground truth examples, label the provided text as: 'deceptive', 'not deceptive', or 'unknown'. On a new line, give a step-by-step breakdown of your reasoning for the decision, citing the ground truth examples where appropriate.
{{text}}

### Assistant:
Label:"""

    SYSTEM_MSG_ZEROSHOT = """You are an AI Assistant with expertise in the detection of digital deception. 

Your approach to deception detection:
Evaluate the query text and categorize it as either 'deceptive', 'not deceptive', or 'unknown' (if it cannot be determined). Do not use any internal or external knowledge you cannot verify and cite.

"""

    USER_INPUT_ZEROSHOT = """### User:
Instruction: Label the provided text as: 'deceptive', 'not deceptive', or 'unknown'. On a new line, give a step-by-step breakdown of your reasoning.
{{text}}

### Assistant:
Label:"""

    SYSTEM_MSG_OLD = """You are an AI Assistant with expertise in the detection of digital deception. You use the following definition of deception: {{definition}}
    
Deceptive texts aim to achieve malicious objectives and include dishonest content such as misinformation, fake news, propaganda, spam, scams, phishing attempts, opinion spam, misinformation and false rumours on twitter and other social media, opinion spam or spam product reviews
    
Deception in text is defined as follows:
{{definition}}

Approach to Detection:
Using only the definition of deception ahd the context of ground truth texts, evaluate the text and categorize it as either deceptive or not deceptive. If the answer cannot be determined, say unknown. DO use definition of deception and context, do NOT use any internal or external knowledge you cannot verify and cite. Give a step-by-step explanation with references to specific ground truth examples and the definition of deception.
Use the following context and the deinition of deception within it to help respond to the instruction from the user.
IF you are not sure of the answer, say I do not know.

Use the following context and the definition of deception to help respond to the instruction from the user:
Context:
{% for doc in documents %}
### User:
Instruction: Relying only on the definition of deception and context, label the text below as: deceptive, not deceptive.
{{doc.content}}
### Assistant:
Label: {{doc.meta['tag']}}
{% endfor %}"""

    USER_INPUT_OLD = """### User:
Instruction: Relying only on the definition of deception and context, label the text below as: deceptive, not deceptive. If you are unsurse of the answer, say unknown. On a new line, explain your reasoning step-by-step, citing the definition of deception where appropriate.
{{text}}

### Assistant:
Label:"""


class MistralInstructPrompt(BasePrompt):
    """Prompt templates using Mistral Instruct [INST] chat format."""

    SYSTEM_MSG ="""<s>[INST] You are an AI Assistant with expertise in the detection of digital deception. 

Deception is defined as follows:
{{definition}}

Your approach to deception detection:
Using the deception definition and the provided ground truth examples, evaluate the query text and categorize it as either 'deceptive', 'not deceptive', or 'unknown' (if it cannot be determined). Explicitly use both the deception definition and the example context in making your decision; do not use any internal or external knowledge that you cannot verify or cite.

Ground Truth Examples:
{% for doc in documents %}
Instruction: Using the deception definition and the provided ground truth examples, label the provided text as: 'deceptive', 'not deceptive', or 'unknown'.
{{doc.content}} [/INST]
Label: {{doc.meta['tag']}}</s>
<s>[INST] {% endfor %}"""

    USER_INPUT = """Instruction: Using the deception definition and the provided ground truth examples, label the provided text as: 'deceptive', 'not deceptive', or 'unknown'. On a new line, give a step-by-step breakdown of your reasoning for the decision, citing the deception definition and the ground truth examples where appropriate.
{{text}} [/INST]
Label:"""

    SYSTEM_MSG_DEFINITIONLESS = """<s>[INST] You are an AI Assistant with expertise in the detection of digital deception. 

Your approach to deception detection:
Using the provided ground truth examples, evaluate the query text and categorize it as either 'deceptive', 'not deceptive', or 'unknown' (if it cannot be determined). Explicitly use the example context in making your decision; do not use any internal or external knowledge that you cannot verify or cite.

Ground Truth Examples:
{% for doc in documents %}
Instruction: Using the provided ground truth examples, label the provided text as: 'deceptive', 'not deceptive', or 'unknown'.
{{doc.content}} [/INST]
Label: {{doc.meta['tag']}}</s>
<s>[INST] {% endfor %}"""

    USER_INPUT_DEFINITIONLESS = """Instruction: Using the provided ground truth examples, label the provided text as: 'deceptive', 'not deceptive', or 'unknown'. On a new line, give a step-by-step breakdown of your reasoning for the decision, citing the ground truth examples where appropriate.
{{text}} [/INST]
Label:"""

    SYSTEM_MSG_ZEROSHOT = """<s>[INST] You are an AI Assistant with expertise in the detection of digital deception. 

Your approach to deception detection:
Evaluate the query text and categorize it as either 'deceptive', 'not deceptive', or 'unknown' (if it cannot be determined). Do not use any internal or external knowledge you cannot verify and cite.

"""

    USER_INPUT_ZEROSHOT = """Instruction: Label the provided text as: 'deceptive', 'not deceptive', or 'unknown'. On a new line, give a step-by-step breakdown of your reasoning.
{{text}} [/INST]
Label:"""

    SYSTEM_MSG_OLD = """<s>[INST] You are an AI Assistant with expertise in the detection of digital deception. You use the following definition of deception: {{definition}}
    
Deceptive texts aim to achieve malicious objectives and include dishonest content such as misinformation, fake news, propaganda, spam, scams, phishing attempts, opinion spam, misinformation and false rumours on twitter and other social media, opinion spam or spam product reviews
    
Deception in text is defined as follows:
{{definition}}

Approach to Detection:
Using only the definition of deception ahd the context of ground truth texts, evaluate the text and categorize it as either deceptive or not deceptive. If the answer cannot be determined, say unknown. DO use definition of deception and context, do NOT use any internal or external knowledge you cannot verify and cite. Give a step-by-step explanation with references to specific ground truth examples and the definition of deception.
Use the following context and the deinition of deception within it to help respond to the instruction from the user.
IF you are not sure of the answer, say I do not know.

Use the following context and the definition of deception to help respond to the instruction from the user:
Context:
{% for doc in documents %}
Instruction: Relying only on the definition of deception and context, label the text below as: deceptive, not deceptive.
{{doc.content}} [/INST]
Label: {{doc.meta['tag']}}</s>
<s>[INST] {% endfor %}"""

    USER_INPUT_OLD = """Instruction: Relying only on the definition of deception and context, label the text below as: deceptive, not deceptive. If you are unsurse of the answer, say unknown. On a new line, explain your reasoning step-by-step, citing the definition of deception where appropriate.
{{text}} [/INST]
Label:"""


class GemmaPrompt(BasePrompt):
    """Prompt templates using Gemma instruction-tuned chat format."""

    SYSTEM_MSG ="""<bos><start_of_turn>user
You are an AI Assistant with expertise in the detection of digital deception. 

Deception is defined as follows:
{{definition}}

Your approach to deception detection:
Using the deception definition and the provided ground truth examples, evaluate the query text and categorize it as either 'deceptive', 'not deceptive', or 'unknown' (if it cannot be determined). Explicitly use both the deception definition and the example context in making your decision; do not use any internal or external knowledge that you cannot verify or cite.

Ground Truth Examples:
{% for doc in documents %}
Instruction: Using the deception definition and the provided ground truth examples, label the provided text as: 'deceptive', 'not deceptive', or 'unknown'.
{{doc.content}}<end_of_turn>
<start_of_turn>model
Label: {{doc.meta['tag']}}<end_of_turn>
<start_of_turn>user
{% endfor %}"""

    USER_INPUT = """Instruction: Using the deception definition and the provided ground truth examples, label the provided text as: 'deceptive', 'not deceptive', or 'unknown'. On a new line, give a step-by-step breakdown of your reasoning for the decision, citing the deception definition and the ground truth examples where appropriate.
{{text}}<end_of_turn>
<start_of_turn>model
Label:"""

    SYSTEM_MSG_DEFINITIONLESS = """<bos><start_of_turn>user
You are an AI Assistant with expertise in the detection of digital deception. 

Your approach to deception detection:
Using the provided ground truth examples, evaluate the query text and categorize it as either 'deceptive', 'not deceptive', or 'unknown' (if it cannot be determined). Explicitly use the example context in making your decision; do not use any internal or external knowledge that you cannot verify or cite.

Ground Truth Examples:
{% for doc in documents %}
Instruction: Using the provided ground truth examples, label the provided text as: 'deceptive', 'not deceptive', or 'unknown'.
{{doc.content}}<end_of_turn>
<start_of_turn>model
Label: {{doc.meta['tag']}}<end_of_turn>
<start_of_turn>user
{% endfor %}"""

    USER_INPUT_DEFINITIONLESS = """Instruction: Using the provided ground truth examples, label the provided text as: 'deceptive', 'not deceptive', or 'unknown'. On a new line, give a step-by-step breakdown of your reasoning for the decision, citing the ground truth examples where appropriate.
{{text}}<end_of_turn>
<start_of_turn>model
Label:"""

    SYSTEM_MSG_ZEROSHOT = """<bos><start_of_turn>user
You are an AI Assistant with expertise in the detection of digital deception. 

Your approach to deception detection:
Evaluate the query text and categorize it as either 'deceptive', 'not deceptive', or 'unknown' (if it cannot be determined). Do not use any internal or external knowledge you cannot verify and cite.

"""

    USER_INPUT_ZEROSHOT = """Instruction: Label the provided text as: 'deceptive', 'not deceptive', or 'unknown'. On a new line, give a step-by-step breakdown of your reasoning.
{{text}}<end_of_turn>
<start_of_turn>model
Label:"""

    SYSTEM_MSG_OLD = """<bos><start_of_turn>user
You are an AI Assistant with expertise in the detection of digital deception. You use the following definition of deception: {{definition}}
    
Deceptive texts aim to achieve malicious objectives and include dishonest content such as misinformation, fake news, propaganda, spam, scams, phishing attempts, opinion spam, misinformation and false rumours on twitter and other social media, opinion spam or spam product reviews
    
Deception in text is defined as follows:
{{definition}}

Approach to Detection:
Using only the definition of deception ahd the context of ground truth texts, evaluate the text and categorize it as either deceptive or not deceptive. If the answer cannot be determined, say unknown. DO use definition of deception and context, do NOT use any internal or external knowledge you cannot verify and cite. Give a step-by-step explanation with references to specific ground truth examples and the definition of deception.
Use the following context and the deinition of deception within it to help respond to the instruction from the user.
IF you are not sure of the answer, say I do not know.

Use the following context and the definition of deception to help respond to the instruction from the user:
Context:
{% for doc in documents %}
Instruction: Relying only on the definition of deception and context, label the text below as: deceptive, not deceptive.
{{doc.content}}<end_of_turn>
<start_of_turn>model
Label: {{doc.meta['tag']}}<end_of_turn>
<start_of_turn>user
{% endfor %}"""

    USER_INPUT_OLD = """Instruction: Relying only on the definition of deception and context, label the text below as: deceptive, not deceptive. If you are unsurse of the answer, say unknown. On a new line, explain your reasoning step-by-step, citing the definition of deception where appropriate.
{{text}}<end_of_turn>
<start_of_turn>model
Label:"""


class Prompt(QwenPrompt):
    """Backward-compatible alias for the default Qwen prompt templates."""


class PromptRegistry:
    """Lookup table for model-family-specific prompt templates."""

    PROMPTS = {
        "qwen": QwenPrompt,
        "qwen3": QwenPrompt,
        "neural_chat": NeuralChatPrompt,
        "mistral_instruct": MistralInstructPrompt,
        "mistral": MistralInstructPrompt,
        "gemma": GemmaPrompt,
        "gemma2": GemmaPrompt,
        "gemma2_it": GemmaPrompt,
    }

    @classmethod
    def get(cls, prompt_profile: str):
        try:
            return cls.PROMPTS[prompt_profile]
        except KeyError as exc:
            known_profiles = ", ".join(sorted(cls.PROMPTS))
            raise ValueError(f"Unknown prompt profile '{prompt_profile}'. Known profiles: {known_profiles}") from exc


class LLM:
    """Global constants for LLM use."""
    # Memory and device configuration
    DEFAULT_MAX_MEMORY = {0: "15GB", "cpu": "9GB"}
    LOW_CPU_MEM_USAGE = True
    USE_CACHE = True

    # Model specification
    DEFAULT_MODEL_KEY = "qwen3"
    DEFAULT_LLM = "Qwen/Qwen3-8B"
    TRUST_REMOTE_CODE = True
    USE_FAST_TOKENIZER = True
    PAD_TOKEN_ID = 0  # fallback only; tokenizer.pad_token_id is used when available
    ATTN_IMPLEMENTATION = "sdpa" # "flash_attention_2", "sdpa", or None
    
    # Quantization settings
    QUANTIZATION_MODE = "4bit" # None, 4bit, 8bit

    # Input parameters
    MAX_INPUT_TOKENS = 8192

    # Generation parameters
    MAX_OUTPUT_TOKENS = 512 # fallback only; per-model max_new_tokens are set in MODEL_PROFILES
    DO_SAMPLE = False
    TEMPERATURE = 0.7 # ignored if DO_SAMPLE=False
    TOP_P = 0.95 # ignored if DO_SAMPLE=False
    TOP_K_TOKENS = 40 # ignored if DO_SAMPLE=False
    REPETITION_PENALTY = 1.05 # fallback; per-model values are in MODEL_PROFILES
    NUM_RETURN_SEQUENCES = 1

    MODEL_PROFILES = {
        "qwen3": {
            "model_name": "Qwen/Qwen3-8B",
            "prompt_profile": "qwen3",
            "max_input_tokens": 8192,
            "max_new_tokens": 2048,
            "attn_implementation": "sdpa",
            "trust_remote_code": True,
            "use_fast_tokenizer": True,
            "pad_token_id": 0,
            "repetition_penalty": 1.05,
        },
        "neural_chat": {
            "model_name": "Intel/neural-chat-7b-v3-3",
            "prompt_profile": "neural_chat",
            "max_input_tokens": 8192,
            "max_new_tokens": 2048,
            "attn_implementation": "sdpa",
            "trust_remote_code": True,
            "use_fast_tokenizer": True,
            "pad_token_id": 0,
            "repetition_penalty": 1.2,
        },
        "mistral_instruct": {
            "model_name": "mistralai/Mistral-7B-Instruct-v0.2",
            "prompt_profile": "mistral_instruct",
            "max_input_tokens": 8192,
            "max_new_tokens": 2048,
            "attn_implementation": "sdpa",
            "trust_remote_code": False,
            "use_fast_tokenizer": True,
            "pad_token_id": 2,
            "repetition_penalty": 1.1,
        },
        "gemma2_it": {
            "model_name": "google/gemma-2-9b-it",
            "prompt_profile": "gemma2_it",
            "max_input_tokens": 8192,
            "max_new_tokens": 2048,
            "attn_implementation": "sdpa",
            "trust_remote_code": False,
            "use_fast_tokenizer": True,
            "pad_token_id": 0,
            "repetition_penalty": 1.1,
        },
    }

    MODEL_ALIASES = {
        "qwen": "qwen3",
        "qwen3": "qwen3",
        "Qwen/Qwen3-8B": "qwen3",
        "neural-chat": "neural_chat",
        "neural_chat": "neural_chat",
        "Intel/neural-chat-7b-v3-3": "neural_chat",
        "mistral": "mistral_instruct",
        "mistral_instruct": "mistral_instruct",
        "mistral-7b-instruct-v0.2": "mistral_instruct",
        "mistralai/Mistral-7B-Instruct-v0.2": "mistral_instruct",
        "gemma": "gemma2_it",
        "gemma2": "gemma2_it",
        "gemma2_it": "gemma2_it",
        "gemma-2-9b-it": "gemma2_it",
        "google/gemma-2-9b-it": "gemma2_it",
    }

    @classmethod
    def get_model_profile(cls, model_key: str) -> dict:
        """Return model defaults for a known model key, alias, or HF model id."""
        canonical_key = cls.MODEL_ALIASES.get(model_key, model_key)
        try:
            return cls.MODEL_PROFILES[canonical_key].copy()
        except KeyError as exc:
            known_models = ", ".join(sorted(cls.MODEL_PROFILES))
            raise ValueError(f"Unknown LLM model key '{model_key}'. Known models: {known_models}") from exc

class Evaluation:
    """Global constants for text/output parsing."""
    # Pattern matching for LLM response parsing
    PARSING_PATTERNS = {
        ": deceptive": "deceptive",
        ": delective": "deceptive",
        ": not deceptive": "non-deceptive",
        "not delective": "non-deceptive",
        "not decessive": "non-deceptive",
        "non deceptive": "non-deceptive",
        "non delective": "non-deceptive",
        "non decessive": "non-deceptive",
        "partially deceptive": "deceptive",
        "probably deceptive": "deceptive",
        "potentially deceptive": "deceptive",
        "possibly deceptive": "deceptive",
        "likely deceptive": "deceptive",
        "somewhat deceptive": "deceptive",
        "deceptive, non-deceptive": "both",
        "non-deceptive, deceptive": "both",
        "figuratively deceptive": "both",
        "figuratively non-deceptive": "both",
    }

    # Close words
    DECEPTIVE_TYPOS = [
        "deceptive", 
        "delective", 
        "decessive"
    ]
    NOT_DECEPTIVE_TYPOS = [
        "not deceptive", 
        "not delective", 
        "not decessive", 
        "undeceptive",
        "un-deceptive",
        "nondeceptive", 
        "non-deceptive"
    ]

    # Qualifiers that signify deceptive
    QUALIFIERS = [
        "partially",
        "potentially",
        "possibly",
        "likely",
        "somewhat",
        "probably"
    ]
    
    # CSV export settings
    DEFAULT_CSV_FILENAME = "llm_responses_log.csv"
    CSV_HEADERS = ["reasoning", "prediction_string", "numerical_score", "correct"]

    LOGGING_HEADERS = [
        "definition", # definition type used (0, 1, 2, 3)
        "format", # prompt format used (new = FRAUD-ICL, old = RADDICL 1.0 prior work)
        "retrieval", # retrieval method used
        "order", # ordering strategy of pipeline
        "indices", # indices of input in the original order
        "domain", # domain of input
        "prompt", # input itself
        "true_label", # input label
        "pred_label", # predicted label, variable is "score"
        "pred_tag", # predicted label as tag, also referred to as "predicted_label"
        "reasoning", # explaination for prediction
        "valid", # whether predicted label is 0, 1
        "correct", # whether predicted label == true label
        "raw_answer", # generated output (with .strip())
        "raw_prompt", # complete augmented prompt sent to the LLM
        "retrieved_examples", # JSON list of retrieved examples used for the query
        "error", # error for parsing, if any
        # --- grounding columns are inserted here by logging_headers() ---
        # The unsuffixed set below belongs to LettuceDetect.PRIMARY_CONTEXT_MODE
        # ("demos"), the mode the paper reports.
        # Additional scored modes append suffixed blocks after it, e.g.
        # grounded_demos_query, hallucinated_fraction_demos_query, ...
        "grounded", # whether reasoning is grounded in the reference context (None if checker not used)
        "hallucination_score", # max confidence of any unsupported span (None if grounded or checker not used)
        "hallucinated_fraction", # fraction of answer chars marked unsupported
        "hallucinated_spans", # JSON list of unsupported span objects
        "grounding_modes", # which context modes were scored, "|"-joined (provenance)
        "seconds" # number of seconds passthrough took
    ]

    @classmethod
    def logging_headers(cls, modes=None):
        """CSV header for a run that scored `modes`.

        The primary mode keeps the unsuffixed grounding columns; every other mode
        contributes a suffixed block immediately after, in the order requested.
        Passing None yields the legacy single-mode header unchanged.
        """
        if not modes:
            return list(cls.LOGGING_HEADERS)
        ld = LettuceDetect
        extra = [
            ld.column_name(f, m)
            for m in modes if m != ld.PRIMARY_CONTEXT_MODE
            for f in ld.GROUNDING_FIELDS
        ]
        if not extra:
            return list(cls.LOGGING_HEADERS)
        out = list(cls.LOGGING_HEADERS)
        anchor = out.index("grounding_modes")
        return out[:anchor] + extra + out[anchor:]

    DOMAIN_HEADERS = [
        "template",
        "order",
        "domain",
        "count",
        "coverage",
        "accuracy",
        "f1",
        "tp",
        "tn",
        "fp",
        "fn",
        "invalid",
        "precision",
        "recall",
        "avg_time"
    ]
