import fraudicl.global_config as global_config
from fraudicl.helper import session_timer, legacy_format_to_fro_tag
from typing import Dict, Any, List, Optional
from pprint import pprint
import json
import re
import torch

_THINKING_TAG_RE = re.compile(r'<think>.*?</think>', re.DOTALL)


def _json_safe(value: Any) -> Any:
    """Convert numpy/scalar values to JSON-serializable Python values."""
    if hasattr(value, "item"):
        return value.item()
    return value


def scored_answer(full_answer: Any) -> str:
    """The exact text grounding is scored over: the full answer, tags stripped.

    The whole answer, not the parsed `reasoning` field: the departure fraction
    is a share of answer characters, so the denominator must be the text the
    detector saw, and `reasoning` omits the verdict sentence. Falls back to the unstripped
    answer when stripping leaves nothing.
    """
    raw = "" if full_answer is None else str(full_answer)
    return _THINKING_TAG_RE.sub("", raw).strip() or raw

class AnswerParser:
    """Parses LLM text outputs to extract prediction labels, reasoning, and scores."""

    def __init__(self) -> None:
        """Initialize parser (stateless, no setup needed)."""
        pass

    def _strip_thinking_tags(self, answer: str) -> str:
        """Remove <think>...</think> blocks from Qwen-style chain-of-thought responses."""
        return _THINKING_TAG_RE.sub('', answer).strip()

    def parse_output(self, answer: str) -> Dict[str, Any]:
        """Extract prediction, reasoning, and numerical score from LLM response text."""
        # Initialize result
        result = {
            "pred": "",
            "reasoning": "",
            "score": "",
            "error": ""
        }

        answer = self._strip_thinking_tags(answer)

        try:
            parsed = self._parse_structured_answer(answer)
            if parsed is not None:
                result.update(parsed)
            else: # manual parsing
                backup_result = self._parse_backup(answer)
                for k in backup_result.keys():
                    result[k] = backup_result[k]
        except (ValueError, IndexError, KeyError) as e:
            result["error"] = f"Parse error: {str(e)}"
            result["pred"] = "PARSER ERROR"
            result["reasoning"] = "PARSER ERROR"

        result["score"] = self._prediction_to_score(result["pred"])

        return result

    def _parse_structured_answer(self, answer: str) -> Optional[Dict[str, Any]]:
        """Parse common label-first formats from model continuations.

        Direct generation strips the prompt from the decoded output. For prompts
        ending in ``Label:``, models such as Gemma often return just
        ``deceptive`` or ``non-deceptive`` on the first generated line.
        """
        cleaned_answer = answer.replace("<end_of_turn>", "").strip()
        if not cleaned_answer:
            return None

        label_match = re.search(
            r"(?im)^\s*(?:\*\*)?\s*(?:label|prediction|answer)\s*(?:\*\*)?\s*:\s*"
            r"(?P<label>non[-\s]?deceptive|not\s+deceptive|deceptive|unknown)\b"
            r"(?P<rest>.*)$",
            cleaned_answer
        )
        if label_match:
            pred = self._clean_prediction(label_match.group("label"))
            remaining_lines = cleaned_answer[label_match.end():].strip()
            same_line_rest = label_match.group("rest").strip()
            reasoning = "\n".join(part for part in (same_line_rest, remaining_lines) if part).strip()
            return {
                "pred": pred,
                "reasoning": self._clean_reasoning(reasoning),
                "error": ""
            }

        first_line, sep, rest = cleaned_answer.partition("\n")
        leading_match = re.match(
            r"^\s*(?:[-*]\s*)?(?:\*\*)?\s*"
            r"(?P<label>non[-\s]?deceptive|not\s+deceptive|deceptive|unknown)"
            r"(?:\s*(?:\*\*)?\s*[.:-]?\s*)$",
            first_line,
            re.IGNORECASE
        )
        if leading_match:
            return {
                "pred": self._clean_prediction(leading_match.group("label")),
                "reasoning": self._clean_reasoning(rest if sep else ""),
                "error": ""
            }

        return None

    def _clean_reasoning(self, reasoning: str) -> str:
        """Remove common generated section labels from extracted reasoning."""
        return re.sub(
            r"^\s*(?:\*\*)?\s*(?:reasoning|explanation)(?:\*\*)?\s*:?\s*",
            "",
            reasoning.strip(),
            flags=re.IGNORECASE
        ).strip()

    def _parse_backup(self, answer: str) -> Dict[str, Any]:
        """error parsing when standard format is not found."""
        # Initialize result
        result = {
            "pred": "",
            "reasoning": "",
            "error": ""
        }

        # Check patterns
        ans = answer.strip().lower()
        found = False
        for k, v in global_config.Evaluation.PARSING_PATTERNS.items():
            if k in ans:
                result["pred"] = v
                result["reasoning"] = answer.strip().split(k, 1)[-1].strip() if len(ans.split(k)) > 1 else answer
                result["error"] = "Manual"
                found = True
                break

        if not found:
            for k in result.keys():
                result[k] = "PARSER ERROR"

        return result

    def _clean_prediction(self, pred: str) -> str:
        """Clean and standardize prediction string by removing artifacts."""
        cleaned = pred.lower().strip().strip("*").strip(".").strip("-").strip()
        # Normalize "non deceptive" variants to "non-deceptive"
        if cleaned in ("non deceptive", "nondeceptive", "not deceptive"):
            cleaned = "non-deceptive"
        return cleaned

    def _prediction_to_score(self, pred: str) -> int:
        """Convert prediction string to numerical score for evaluation."""
        # Standardize "<qualifier> deceptive" (e.g. "likely deceptive")
        if "deceptive" in pred and any(qualifier in pred for qualifier in global_config.Evaluation.QUALIFIERS):
            pred = "deceptive"

        # Reverse lookup for tag/score
        try:
            score = next(k for k, v in global_config.Database.LABEL_MAPPING.items() if v == pred)
        except StopIteration:
            score = -1

        return score

class HallucinationChecker:
    """Checks whether LLM reasoning is grounded in the retrieved context documents.

    Uses LettuceDetect (transformer-based token classifier) to identify spans
    in the LLM answer that are not supported by the retrieved passages.

    The detector is loaded lazily on the first call to check(), so it does not
    consume GPU memory until needed.
    """

    def __init__(
        self,
        model_path: str = global_config.LettuceDetect.DEFAULT_MODEL,
        device: Optional[str] = None,
        context_mode: str = global_config.LettuceDetect.DEFAULT_CONTEXT_MODE
    ) -> None:
        """
        Args:
            model_path:   HuggingFace model ID or local path.
                          Defaults to the faster base model.
            device:       Device string ("cuda:0", "cpu", etc.).
                          None lets LettuceDetect auto-select.
            context_mode: Which prompt pieces count as supporting evidence
                          (see global_config.LettuceDetect.CONTEXT_MODES).
                          Defaults to "demos", which reproduces the paper's
                          numbers; "demos+query" also counts the text being
                          classified as evidence. Absolute levels are not
                          comparable across modes; compare within a mode.
        """
        if context_mode not in global_config.LettuceDetect.CONTEXT_MODES:
            raise ValueError(
                f"context_mode must be one of {global_config.LettuceDetect.CONTEXT_MODES}, "
                f"got {context_mode!r}"
            )
        self.model_path = model_path
        self.device = device
        self.context_mode = context_mode
        self._detector = None

    @staticmethod
    def build_context(
        context_docs: List[str],
        question: Optional[str] = None,
        definition: Optional[str] = None,
        context_mode: str = global_config.LettuceDetect.DEFAULT_CONTEXT_MODE
    ) -> List[str]:
        """Assemble the evidence list handed to LettuceDetect for one response.

        LettuceDetect's ``question`` argument steers span detection but is not
        itself treated as support, so the query must be appended to ``context``
        for query-grounded rationale text to count as supported.
        """
        cfg = global_config.LettuceDetect
        context = list(context_docs)
        if context_mode in ("demos+query", "demos+query+def") and question:
            context.append(f"{cfg.QUERY_PREFIX}{question.strip()}")
        if context_mode == "demos+query+def" and definition:
            context.append(f"{cfg.DEFINITION_PREFIX}{definition.strip()}")
        return context

    def _load(self) -> None:
        """Lazy-load the LettuceDetect detector on first call."""
        if self._detector is not None:
            return
        import os
        os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
        # Imported lazily so the detector costs no memory until first use.
        try:
            from lettucedetect.models.inference import HallucinationDetector
        except ImportError as exc:
            raise ImportError(
                "LettuceDetect is required for hallucination grounding checks. "
                "Install project dependencies first (see README Install: `uv sync`)."
            ) from exc
        kwargs = {"method": "transformer", "model_path": self.model_path}
        if self.device is not None:
            kwargs["device"] = self.device
        self._detector = HallucinationDetector(**kwargs)

    def check(
        self,
        reasoning: str,
        context_docs: List[str],
        question: Optional[str] = None,
        definition: Optional[str] = None,
        context_mode: Optional[str] = None
    ) -> Dict[str, Any]:
        """Check if reasoning is grounded in the retrieved documents.

        Args:
            reasoning:    The full LLM answer to inspect (after thinking-tag stripping).
            context_docs: List of retrieved passage strings.
            question:     The original input text. Passed as LettuceDetect's question
                          and, unless context_mode is "demos", also added to the
                          context so query-grounded rationale counts as supported.
            definition:   The deception definition shown in the prompt; only used
                          when context_mode is "demos+query+def".
            context_mode: Overrides the instance's context_mode for this call.

        Returns:
            Dict with keys:
                grounded             (bool)  True if no hallucinated spans found.
                hallucination_score  (float) Max span confidence; 0.0 if grounded.
                hallucinated_fraction(float) Fraction of answer chars hallucinated.
                hallucinated_spans   (str)   JSON list of span dicts.
        """
        self._load()
        mode = context_mode or self.context_mode
        try:
            spans = self._detector.predict(
                context=self.build_context(context_docs, question, definition, mode),
                answer=reasoning,
                question=question,
                output_format="spans"
            )
            grounded = len(spans) == 0
            hallucination_score = float(max((s["confidence"] for s in spans), default=0.0))
            total_chars = len(reasoning)
            hallucinated_chars = sum(s["end"] - s["start"] for s in spans)
            hallucinated_fraction = hallucinated_chars / total_chars if total_chars > 0 else 0.0
            hallucinated_spans = json.dumps(spans)
        except Exception as e:
            grounded = None
            hallucination_score = None
            hallucinated_fraction = None
            hallucinated_spans = f"ERROR: {e}"

        return {
            "grounded": grounded,
            "hallucination_score": hallucination_score,
            "hallucinated_fraction": hallucinated_fraction,
            "hallucinated_spans": hallucinated_spans
        }

    def check_all(
        self,
        reasoning: str,
        context_docs: List[str],
        question: Optional[str] = None,
        definition: Optional[str] = None,
        modes: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Score one response under several context modes in a single pass.

        The detector loads once and each distinct context costs one forward pass.

        Returns a flat dict keyed by column name, with the primary mode owning the
        unsuffixed fields (see ``LettuceDetect.column_name``). Modes are scored in
        the order given, and identical context assemblies are computed once and
        reused -- notably ``demos+query+def`` collapses onto ``demos+query`` when
        the prompt carried no definition text (definition_type -1/0), so those
        rows are not scored twice.
        """
        cfg = global_config.LettuceDetect
        modes = list(modes) if modes else [cfg.PRIMARY_CONTEXT_MODE]

        out: Dict[str, Any] = {}
        seen: Dict[tuple, Dict[str, Any]] = {}
        for mode in modes:
            context = self.build_context(context_docs, question, definition, mode)
            key = tuple(context)
            if key not in seen:
                seen[key] = self.check(
                    reasoning=reasoning,
                    context_docs=context_docs,
                    question=question,
                    definition=definition,
                    context_mode=mode,
                )
            result = seen[key]
            for field in cfg.GROUNDING_FIELDS:
                out[cfg.column_name(field, mode)] = result[field]
        out["grounding_modes"] = "|".join(modes)
        return out

    def unload(self) -> None:
        """Release the detector from memory."""
        self._detector = None
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

class EvaluationManager:
    """Runs text through the full RAG pipeline and parses results for evaluation."""

    def __init__(self, pipeline) -> None:
        """Initialize with a PipelineManager instance."""
        self.pipeline = pipeline
        self.answer_parser = AnswerParser()

    def passthrough(self,
                    text: str,
                    true_label: int,
                    definition_type: int,
                    use_legacy_format: bool,
                    order_strategy: str,
                    retrieval_strategy: str,
                    top_k: int = global_config.Retrieval.TOP_K,
                    hallucination_checker: Optional[HallucinationChecker] = None,
                    grounding_modes: Optional[List[str]] = None,
                    verbose: bool = False
                    ) -> Dict[str, Any]:
        """Run text through pipeline, parse output, and return structured result.

        grounding_modes selects which LettuceDetect reference contexts to score
        against (see global_config.LettuceDetect.CONTEXT_MODES):
          "demos"           -- retrieved demonstrations only; reproduces the
                               published demonstration-relative departure.
          "demos+query"     -- adds the evaluated text, so observations about the
                               text under judgment can count as supported.
          "demos+query+def" -- also adds the deception definition, matching what
                               the system prompt told the model to reason from.
                               Collapses onto "demos+query" for definition types
                               -1/0, which carry no definition text.
        Results for the primary mode occupy the unsuffixed grounding keys; other
        modes are returned under `<field>_<slug>` keys. Defaults to all three.
        """
        # Instantiate result
        result = {
            "full_answer": "",
            "indices": [],
            "pred": "",
            "reasoning": "",
            "score": -1,
            "error": False,
            "definition_type": definition_type,
            "use_legacy_format": use_legacy_format,
            "retrieval_strategy": retrieval_strategy,
            "order_strategy": order_strategy,
            "retrieved_docs": [],
            "raw_prompt": "",
            "retrieved_examples": [],
        }
        # Pre-declare every grounding column for the requested modes so rows that
        # skip scoring (no checker, or zero retrieved docs) still carry the full
        # set of keys and line up with the CSV header.
        _modes = grounding_modes or [global_config.LettuceDetect.PRIMARY_CONTEXT_MODE]
        for _col in global_config.LettuceDetect.columns_for(_modes):
            result[_col] = None
        result["grounding_modes"] = None

        if verbose:
            print(f"Query:")
            pprint(text)
            print("-"*30)
            session_timer.start_time("recent-passthrough")

        # Run pipeline + parse
        analyzed = self._analyze_text(text, definition_type, use_legacy_format, order_strategy, retrieval_strategy, top_k)
        for k in analyzed.keys():
            result[k] = analyzed[k]

        # Run hallucination grounding check if checker is provided
        if hallucination_checker is not None and result["retrieved_docs"]:
            hall = hallucination_checker.check_all(
                reasoning=scored_answer(result["full_answer"]),
                context_docs=result["retrieved_docs"],
                question=text,
                definition=self._definition_text(definition_type),
                modes=grounding_modes,
            )
            result.update(hall)

        if verbose:
            session_timer.stop_time("recent-passthrough")
            print(f"System Prompt: {legacy_format_to_fro_tag(use_legacy_format).title()} Definition {definition_type}")
            print(f"Retrieval Strategy: {retrieval_strategy}")
            print(f"Order Strategy: {order_strategy} - {result["indices"]}")
            print("")
            print(f"Correct: {str(analyzed["score"] == true_label).upper()}")
            print(f"Predicted Label: {analyzed["score"]} ({analyzed["pred"]})")
            print(f"True Label: {true_label}")
            print("")
            print("Reasoning:")
            pprint(f"{analyzed["reasoning"]}")
            print("")
            print("Error:")
            pprint(f"{analyzed["error"]}")
            if hallucination_checker is not None:
                hall_score = result['hallucination_score']
                hall_frac = result['hallucinated_fraction']
                if hall_score is not None:
                    print(f"Grounded: {result['grounded']}  |  Score: {hall_score:.4f}  |  Fraction: {hall_frac:.4f}")
                else:
                    print(f"Grounded: {result['grounded']}  |  Score: N/A  |  Fraction: N/A")
            print("-"*30)
            session_timer.report_time("recent-passthrough")

        return result

    def _definition_text(self, definition_type: int) -> Optional[str]:
        """Return the deception definition injected for this definition_type.

        Mirrors PromptManager._generate_prompt_template: zero-shot (-1) and
        definitionless (0) prompts carry no definition text.
        """
        if definition_type in (-1, 0):
            return None
        prompter = getattr(getattr(self.pipeline, "config", None), "prompt_template", None)
        if prompter is None:
            try:
                prompter = self.pipeline.pipeline.get_component("fraudicl_prompt_selector")
            except Exception:
                prompter = None
        prompt_class = getattr(prompter, "prompt_class", global_config.Prompt)
        return getattr(prompt_class, f"DEFINITION_{definition_type}", None)

    def _analyze_text(self,
                      text: str,
                      definition_type: int,
                      use_legacy_format: bool,
                      order_strategy: str,
                      retrieval_strategy: str,
                      top_k: int = global_config.Retrieval.TOP_K
                      ) -> Dict[str, Any]:
        """Given text, parse through it to extract info."""
        text = text.strip()

        with torch.inference_mode():
            generated = self.pipeline.pipeline.run({
                "fraudicl_text_embedder": {"text": text},
                "fraudicl_retriever": {
                    "retrieval_strategy": retrieval_strategy,
                    "top_k": top_k
                },
                "fraudicl_orderer": {
                    "order_strategy": order_strategy,
                    "definition_type": definition_type,
                    "use_legacy_format": use_legacy_format,
                    "test_text": text,
                },
                "fraudicl_prompt_selector": {
                    "text": text,
                    "definition_type": definition_type,
                    "use_legacy_format": use_legacy_format
                }
            })
            llm_output = generated["fraudicl_llm"]
            full_answer = llm_output["replies"][0].strip()
            raw_prompt = llm_output.get("raw_prompt", "")

        # Extract retrieved doc contents for grounding check.
        # Use `retrieved_documents` (unconnected output): `documents` is consumed by the orderer
        # and may be absent from pipeline.run() output in some Haystack versions.
        retrieved_documents = generated.get("fraudicl_retriever", {}).get("retrieved_documents", [])
        retrieved_docs = [doc.content for doc in retrieved_documents if doc.content]
        retrieved_examples = [
            {
                "content": doc.content,
                "label": _json_safe(doc.meta.get("label")) if doc.meta else None,
                "tag": doc.meta.get("tag") if doc.meta else None,
                "domain": doc.meta.get("domain") if doc.meta else None,
                "score": _json_safe(doc.score),
            }
            for doc in retrieved_documents
        ]

        # Parse answer
        results = self.answer_parser.parse_output(full_answer)
        results["full_answer"] = full_answer
        results["indices"] = generated.get("fraudicl_orderer", {}).get("indices", [])
        results["retrieved_docs"] = retrieved_docs
        results["raw_prompt"] = raw_prompt
        results["retrieved_examples"] = retrieved_examples

        return results

    def passthrough_batch(self,
                          texts: List[str],
                          true_labels: List[Any],
                          definition_type: int,
                          use_legacy_format: bool,
                          order_strategy: str,
                          retrieval_strategy: str,
                          top_k: int = global_config.Retrieval.TOP_K,
                          hallucination_checker: Optional[HallucinationChecker] = None,
                          grounding_modes: Optional[List[str]] = None,
                          prepare_hook=None,
                          ) -> List[Dict[str, Any]]:
        """passthrough() for several rows, generating them in one batched pass.

        Retrieval, ordering and prompt assembly stay per row -- only the final
        generation is batched, which is the step that leaves the GPU at ~13%
        utilisation when run one sequence at a time. Results are not guaranteed to equal
        calling passthrough() per row (see --batch-size in main.py and
        Scripts/verify_batched_generation.py).

        prepare_hook, if given, is called with each row's index position in `texts`
        immediately before that row is prepared, so callers can reseed per-row RNG
        at the same point in the sequence the unbatched loop would have.
        """
        modes = grounding_modes or [global_config.LettuceDetect.PRIMARY_CONTEXT_MODE]

        prepared_rows = []
        for position, text in enumerate(texts):
            if prepare_hook is not None:
                prepare_hook(position)
            prepared_rows.append(
                self._prepare_text(
                    text, definition_type, use_legacy_format,
                    order_strategy, retrieval_strategy, top_k,
                )
            )

        with torch.inference_mode():
            generated = self.pipeline.generator.run_batch([p["prompt"] for p in prepared_rows])

        results = []
        for text, true_label, prepared, gen in zip(texts, true_labels, prepared_rows, generated):
            result = {
                "full_answer": "",
                "indices": [],
                "pred": "",
                "reasoning": "",
                "score": -1,
                "error": False,
                "definition_type": definition_type,
                "use_legacy_format": use_legacy_format,
                "retrieval_strategy": retrieval_strategy,
                "order_strategy": order_strategy,
                "retrieved_docs": [],
                "raw_prompt": "",
                "retrieved_examples": [],
            }
            for col in global_config.LettuceDetect.columns_for(modes):
                result[col] = None
            result["grounding_modes"] = None

            result.update(
                self._finish_prepared(prepared, gen["replies"][0], self.answer_parser)
            )

            if hallucination_checker is not None and result["retrieved_docs"]:
                result.update(
                    hallucination_checker.check_all(
                        reasoning=scored_answer(result["full_answer"]),
                        context_docs=result["retrieved_docs"],
                        question=text.strip(),
                        definition=self._definition_text(definition_type),
                        modes=grounding_modes,
                    )
                )
            results.append(result)
        return results

    def _prepare_text(self,
                      text: str,
                      definition_type: int,
                      use_legacy_format: bool,
                      order_strategy: str,
                      retrieval_strategy: str,
                      top_k: int = global_config.Retrieval.TOP_K
                      ) -> Dict[str, Any]:
        """Everything _analyze_text does EXCEPT the final generation.

        Runs embed -> retrieve -> order -> prompt and returns the assembled prompt
        plus the retrieval/ordering context the row needs. Splitting the pipeline
        here is what lets several rows' prompts be generated in one batched forward
        pass; `demo_order` still calls the LLM inside the orderer,
        so its per-instance order selection is unchanged.
        """
        text = text.strip()

        generator = self.pipeline.generator
        generator.prepare_only = True
        try:
            with torch.inference_mode():
                prepared = self.pipeline.pipeline.run({
                    "fraudicl_text_embedder": {"text": text},
                    "fraudicl_retriever": {
                        "retrieval_strategy": retrieval_strategy,
                        "top_k": top_k,
                    },
                    "fraudicl_orderer": {
                        "order_strategy": order_strategy,
                        "definition_type": definition_type,
                        "use_legacy_format": use_legacy_format,
                        "test_text": text,
                    },
                    "fraudicl_prompt_selector": {
                        "text": text,
                        "definition_type": definition_type,
                        "use_legacy_format": use_legacy_format,
                    },
                })
        finally:
            generator.prepare_only = False

        # The generator ran in prepare_only mode, so it echoed the prompt back
        # instead of generating; raw_prompt is the assembled prompt.
        retrieved_documents = prepared.get("fraudicl_retriever", {}).get("retrieved_documents", [])
        return {
            "prompt": prepared["fraudicl_llm"]["raw_prompt"],
            "indices": prepared.get("fraudicl_orderer", {}).get("indices", []),
            "retrieved_documents": retrieved_documents,
        }

    @staticmethod
    def _finish_prepared(prepared: Dict[str, Any], full_answer: str, parser) -> Dict[str, Any]:
        """Assemble a result dict from a prepared row and its generated answer."""
        retrieved_documents = prepared["retrieved_documents"]
        retrieved_docs = [doc.content for doc in retrieved_documents if doc.content]
        retrieved_examples = [
            {
                "content": doc.content,
                "label": _json_safe(doc.meta.get("label")) if doc.meta else None,
                "tag": doc.meta.get("tag") if doc.meta else None,
                "domain": doc.meta.get("domain") if doc.meta else None,
                "score": _json_safe(doc.score),
            }
            for doc in retrieved_documents
        ]
        full_answer = full_answer.strip()
        results = parser.parse_output(full_answer)
        results["full_answer"] = full_answer
        results["indices"] = prepared["indices"]
        results["retrieved_docs"] = retrieved_docs
        results["raw_prompt"] = prepared["prompt"]
        results["retrieved_examples"] = retrieved_examples
        return results
