import fraudicl.global_config as global_config
from typing import List, Dict, Any, Union, Optional, Type
from jinja2 import Template
from haystack import component, default_to_dict, default_from_dict
from haystack.dataclasses import Document

@component
class PromptManager:
    """Manages prompt template generation and formatting for deception detection."""

    def __init__(self, prompt_profile: str = "qwen3", prompt_class: Optional[Type[global_config.BasePrompt]] = None) -> None:
        """Initialize with empty caches for templates and compiled Jinja objects."""
        self.prompt_profile = prompt_profile
        self.prompt_class = prompt_class or global_config.PromptRegistry.get(prompt_profile)
        self.format_combinations: Dict[str, str] = {}
        self._compiled_templates: Dict[str, Template] = {}  # Cache compiled Jinja templates

    def _get_cache_key(self, definition_type: int, use_legacy_format: Union[bool, int]) -> str:
        """Generate cache key for template lookup."""
        if definition_type == -1:
            return "def_zeroshot"  # No legacy variant for zero-shot mode
        if definition_type == 0:
            return "def_0_new"  # No legacy variant for definition-less mode
        tag = "old" if use_legacy_format else "new"
        return f"def_{definition_type}_{tag}"

    def _generate_prompt_template(self, definition_type: int, use_legacy_format: Union[bool, int]) -> str:
        """Generate and cache the finalized prompt template string."""
        cache_key = self._get_cache_key(definition_type, use_legacy_format)

        if cache_key not in self.format_combinations:
            # Determine which prompts to use
            if definition_type == -1:
                dict_system_prompt = self.prompt_class.SYSTEM_MSG_ZEROSHOT
                dict_user_input = self.prompt_class.USER_INPUT_ZEROSHOT
            elif definition_type == 0:
                dict_system_prompt = self.prompt_class.SYSTEM_MSG_DEFINITIONLESS
                dict_user_input = self.prompt_class.USER_INPUT_DEFINITIONLESS
            elif use_legacy_format:
                dict_system_prompt = self.prompt_class.SYSTEM_MSG_OLD
                dict_user_input = self.prompt_class.USER_INPUT_OLD
            else:
                dict_system_prompt = self.prompt_class.SYSTEM_MSG
                dict_user_input = self.prompt_class.USER_INPUT

            # Insert definition if needed (not applicable for zero-shot or definitionless)
            if definition_type not in (-1, 0):
                full_definition = getattr(self.prompt_class, f"DEFINITION_{definition_type}")
                dict_system_prompt = dict_system_prompt.replace("{{definition}}", full_definition)

            # Format and cache
            final_prompt = dict_system_prompt + dict_user_input
            self.format_combinations[cache_key] = final_prompt

        return self.format_combinations[cache_key]

    def _get_compiled_template(self, definition_type: int, use_legacy_format: Union[bool, int]) -> Template:
        """Get or create cached compiled Jinja template (avoids repeated compilation)."""
        cache_key = self._get_cache_key(definition_type, use_legacy_format)

        if cache_key not in self._compiled_templates:
            template_str = self._generate_prompt_template(definition_type, use_legacy_format)
            self._compiled_templates[cache_key] = Template(template_str)

        return self._compiled_templates[cache_key]

    @component.output_types(prompt=str)
    def run(self, definition_type: int, use_legacy_format: Union[bool, int], top_k_examples: List[Document], text: str) -> Dict[str, str]:
        """Build prompt by rendering cached template with provided inputs."""
        jinja_template = self._get_compiled_template(definition_type, use_legacy_format)
        return {"prompt": jinja_template.render(text=text, documents=top_k_examples)}
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize component to dictionary."""
        return default_to_dict(self, prompt_profile=self.prompt_profile)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PromptManager":
        """Deserialize component from dictionary."""
        return default_from_dict(cls, data)
