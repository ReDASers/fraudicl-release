"""Check whether batched generation reproduces single-prompt generation exactly.

Greedy decoding (``DO_SAMPLE = False``) gives each prompt one continuation in exact
arithmetic, but padded-batch logits can differ numerically (notably with 4-bit weights
and a repetition penalty), so equivalence has to be measured. Padding is the other
failure mode: decoder-only models must
be LEFT-padded, or short prompts continue from pad tokens and degrade in ways that
look like plausible output rather than an error.

This script generates the same prompts both ways and compares the decoded text. Run
it against a real backbone before trusting ``--batch-size`` on a long run, and after
any change to padding, truncation, or generation kwargs.

Exit status is nonzero if any prompt differs, so it can gate a run in a shell script.

Usage:
    python Scripts/verify_batched_generation.py --model gemma --batch-size 8
    python Scripts/verify_batched_generation.py --model mistral_instruct \
        --batch-size 16 --max-new-tokens 128   # faster smoke test
"""
from __future__ import annotations

import argparse
import sys

from fraudicl.llmmanager import LLMConfig, LLMManager
from fraudicl.pipelinemanager import DirectLLMGenerator

# Uneven lengths: padding effects only appear when prompts differ in size,
# and the shortest prompt in a batch is the one that breaks first.
PROMPTS = [
    "Answer in one sentence: is a message demanding an urgent gift-card payment deceptive?",
    "Reply briefly.",
    "Consider the following: a seller lists an item, accepts payment, and never ships it. "
    "In two sentences, explain whether this is deceptive and why the buyer's evidence matters.",
    "Say yes or no: is an unsolicited prize notification requesting bank details suspicious?",
    "Summarise, in one short paragraph, why urgency is a common signal in fraudulent "
    "messages, and name one counterexample where urgency is legitimate.",
    "Hi.",
    "List two reasons a product review might be fake.",
    "Explain in one sentence what makes a phishing email convincing.",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="mistral_instruct",
                        help="Model key from global_config.LLM.MODEL_PROFILES.")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Prompts per batched call.")
    parser.add_argument("--quantization", default="4bit",
                        help="Quantization mode, matching the run under test.")
    parser.add_argument("--max-new-tokens", type=int, default=None,
                        help="Override generation length to make the check quicker.")
    parser.add_argument("--prompts", type=int, default=len(PROMPTS),
                        help="How many of the built-in prompts to use.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    prompts = PROMPTS[: args.prompts]

    config = LLMConfig(model_key=args.model, quantization_mode=args.quantization)
    if args.max_new_tokens is not None:
        config.max_new_tokens = args.max_new_tokens

    llm = LLMManager(config=config)
    llm.load_model(quantization_mode=args.quantization, from_hub=True)
    generator = DirectLLMGenerator(llm, llm._prepare_generation_kwargs())

    print("model: {} | batch size: {} | prompts: {}".format(
        config.model_name, args.batch_size, len(prompts)))

    print("\n--- single-prompt baseline ---")
    singles = []
    for i, prompt in enumerate(prompts, start=1):
        singles.append(generator.run(prompt)["replies"][0])
        print("  {}/{}".format(i, len(prompts)), flush=True)

    print("\n--- batched ---")
    batched = []
    for start in range(0, len(prompts), args.batch_size):
        chunk = prompts[start : start + args.batch_size]
        batched.extend(r["replies"][0] for r in generator.run_batch(chunk))
        print("  {}/{}".format(min(start + args.batch_size, len(prompts)), len(prompts)),
              flush=True)

    print("\n=== comparison ===")
    mismatches = 0
    for i, (one, many) in enumerate(zip(singles, batched)):
        if one == many:
            print("[{}] match ({} chars)".format(i, len(one)))
            continue
        mismatches += 1
        print("[{}] MISMATCH".format(i))
        print("     prompt  : {!r}".format(prompts[i][:80]))
        print("     single  : {!r}".format(one[:160]))
        print("     batched : {!r}".format(many[:160]))

    if mismatches:
        print("\n{} of {} prompts differ -- batched generation is NOT equivalent.".format(
            mismatches, len(prompts)))
        print("Do not enable --batch-size until this passes.")
        return 1

    print("\nAll {} prompts identical. Batched generation is equivalent.".format(len(prompts)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
