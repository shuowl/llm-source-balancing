#!/usr/bin/env python3
"""
generate_with_vllm.py - Generate text completions using vLLM with tensor parallelism.

Uses a single vLLM LLM instance, optionally TP-sharded across multiple GPUs
via tensor_parallel_size.

Example:
    python core/generate_with_vllm.py \
        --experiment_name csqa__qwen3_1_7br__d1nu1nin__nocot \
        --probe_variants bare,upos \
        --eval_jsonl data/processed_datasets/csqa_default_split/test.jsonl \
        --tp_size 2
"""

import argparse
import json
import os
import pathlib
import sys
from typing import Dict, List, Optional

# Add project root to path
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from core.exp_name import parse_experiment_name
from core.prompt_style import build_chat_prompt, load_tier_sentences
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


def extract_thinking_with_tags(completion: str) -> str:
    """Extract everything from start up to and including the last </think> tag."""
    last_think_end = completion.rfind("</think>")
    if last_think_end != -1:
        return completion[:last_think_end + len("</think>")]
    return completion


def extract_answer_from_completion(completion: str, reasoning_mode: bool = False) -> str:
    """Extract the answer letter from a completion.

    For normal mode: look for the first standalone A/B/C/D/E.
    For reasoning mode: look for 'Answer:' after the last </think> tag.
    """
    if reasoning_mode:
        think_end = completion.find("</think>")
        if think_end != -1:
            after_think = completion[think_end + len("</think>"):]
            answer_pos = after_think.find("Answer:")
            if answer_pos != -1:
                after_answer = after_think[answer_pos + len("Answer:"):].strip()
                if after_answer and after_answer[0].upper() in "ABCDE":
                    return after_answer[0].upper()

    completion_upper = completion.upper()
    for char in "ABCDE":
        if char in completion_upper:
            idx = completion_upper.index(char)
            if idx == 0 or not completion_upper[idx - 1].isalpha():
                if idx == len(completion_upper) - 1 or not completion_upper[idx + 1].isalpha():
                    return char
    return "UNKNOWN"


def main():
    parser = argparse.ArgumentParser(description="Generate text completions using vLLM (TP only)")
    parser.add_argument("--experiment_name", required=True)
    parser.add_argument("--probe_variants", required=True,
                       help="Comma-separated list of probe variants to process")
    parser.add_argument("--eval_jsonl", required=True)
    parser.add_argument("--output_jsonl", type=str,
                       help="Output path (default: results/{experiment_name}/{probe_variant}_vllm_reasoning.jsonl)")
    parser.add_argument("--canonical_wrong_jsonl", type=str,
                       help="Path to canonical wrong answers (required for non-bare variants)")
    parser.add_argument("--tp_size", type=int, default=1,
                       help="Tensor parallel size (number of GPUs to shard the model across)")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--num_completions", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--reasoning_generation_mode", type=str, default="noncommittal",
                       choices=["full", "noncommittal"])

    args = parser.parse_args()

    import torch
    tp_size = args.tp_size
    num_gpus_available = torch.cuda.device_count()
    print(f"Configuration:")
    print(f"  Tensor Parallel Size: {tp_size}")
    print(f"  GPUs available: {num_gpus_available}")
    if tp_size > num_gpus_available:
        raise ValueError(f"Not enough GPUs: need {tp_size}, only {num_gpus_available} available")

    # Disable NCCL P2P to avoid hanging issues with tensor parallelism
    if tp_size > 1:
        os.environ["NCCL_P2P_DISABLE"] = "1"

    exp = parse_experiment_name(args.experiment_name)

    probe_variants = [pv.strip() for pv in args.probe_variants.split(",")]
    valid_variants = ["bare", "upos", "uneg", "dpos", "dneg",
                     "updp", "updn", "undp", "undn",
                     "dpup", "dpun", "dnup", "dnun"]
    for pv in probe_variants:
        if pv not in valid_variants:
            raise ValueError(f"Invalid probe variant: {pv}")

    if args.output_jsonl and len(probe_variants) > 1:
        raise ValueError("Cannot specify --output_jsonl with multiple probe variants. "
                        "Output files will be generated automatically for each variant.")

    if exp.reasoning_mode and not exp.model_key.startswith('qwen3'):
        raise ValueError("Reasoning mode is only supported for Qwen3 family models")
    if exp.use_cot:
        raise ValueError("COT evaluation is not supported")

    # Load evaluation data
    eval_data = []
    with open(args.eval_jsonl) as f:
        for line in f:
            eval_data.append(json.loads(line))
    print(f"Loaded {len(eval_data)} evaluation examples")

    # Load canonical wrong answers (required for non-bare variants)
    non_bare_variants = [pv for pv in probe_variants if pv != "bare"]
    canonical_wrong_map = {}
    if non_bare_variants:
        if not args.canonical_wrong_jsonl:
            canonical_wrong_path = ROOT / "results" / args.experiment_name / "canonical_wrong.jsonl"
        else:
            canonical_wrong_path = pathlib.Path(args.canonical_wrong_jsonl)
        if canonical_wrong_path.exists():
            with open(canonical_wrong_path) as f:
                for line in f:
                    data = json.loads(line)
                    canonical_wrong_map[data["qid"]] = data["canonical_wrong"]
            print(f"Loaded {len(canonical_wrong_map)} canonical wrong answers")
        else:
            raise ValueError(f"Canonical wrong answers required but not found at {canonical_wrong_path}")

    # Load tier sentences
    tier_sentences_data = None
    if non_bare_variants:
        tier_sentences_data = load_tier_sentences(
            exp.dataset, exp.model_key, exp.instruction, exp.use_cot, exp.reasoning_mode
        )
        if not tier_sentences_data:
            raise ValueError("Tier sentences not found. Run dataset_tier_generator.py first.")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(exp.hf_model_id, trust_remote_code=True)

    # Initialize vLLM (single replica, optionally TP-sharded)
    llm = LLM(
        model=exp.hf_model_id,
        tensor_parallel_size=tp_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=args.max_tokens + 2048,
        disable_log_stats=True,
        enforce_eager=True,
    )

    # Sampling params
    if exp.reasoning_mode and exp.model_key.startswith('qwen3'):
        temperature = args.temperature if args.temperature is not None else 0.6
        top_p = args.top_p if args.top_p is not None else 0.95
        top_k = args.top_k if args.top_k is not None else 20
        if temperature == 0.0:
            temperature = 0.6
        sampling_params = SamplingParams(
            temperature=temperature, top_p=top_p, top_k=top_k,
            max_tokens=args.max_tokens, n=args.num_completions,
        )
    else:
        temperature = args.temperature if args.temperature is not None else 0.0
        top_p = args.top_p if args.top_p is not None else 1.0
        top_k = args.top_k if args.top_k is not None else -1
        sampling_params = SamplingParams(
            temperature=temperature, top_p=top_p,
            top_k=top_k if top_k > 0 else -1,
            max_tokens=args.max_tokens, n=args.num_completions,
        )

    # Pre-parse eval data into a common format
    all_prompts_data = []
    for idx, row in enumerate(eval_data):
        qid = row["id"]
        question = row["question"]
        if isinstance(row["choices"], dict) and "text" in row["choices"]:
            choices = row["choices"]["text"]
        else:
            choices = row["choices"]
        if "answerKey" in row:
            gold_letter = row["answerKey"]
        else:
            gold_text = choices[row["choices"]["label"].index(row["answerKey"])]
            gold_index = choices.index(gold_text)
            gold_letter = chr(ord('A') + gold_index)
        all_prompts_data.append({
            "qid": qid,
            "question": question,
            "choices": choices,
            "gold": gold_letter,
            "original_index": idx,
        })

    # Process each variant
    for probe_variant in probe_variants:
        print(f"\nProcessing variant: {probe_variant}")

        prompts = []
        choices_list = []
        metadata_list = []
        original_indices = []

        for prompt_data in all_prompts_data:
            qid = prompt_data["qid"]

            if probe_variant == "bare":
                wrong_letter = ""
            else:
                wrong_letter = canonical_wrong_map.get(qid, "")
                if not wrong_letter:
                    print(f"Skipping qid {qid} - no canonical wrong answer")
                    continue

            question_tier_sentences = tier_sentences_data.get(qid) if tier_sentences_data else None
            noncommittal = exp.reasoning_mode and args.reasoning_generation_mode == "noncommittal"

            chat_prompt, pure_prompt, system_prompt = build_chat_prompt(
                tokenizer=tokenizer,
                question=prompt_data["question"],
                answer_choices=prompt_data["choices"],
                exp=exp,
                probe_variant=probe_variant,
                gold_answer=prompt_data["gold"],
                wrong_answer=wrong_letter,
                tier_sentences=question_tier_sentences,
                noncommittal_reasoning=noncommittal,
            )
            prompts.append(chat_prompt)
            choices_list.append(prompt_data["choices"])
            metadata_list.append({
                "qid": qid,
                "gold": prompt_data["gold"],
                "user_prompt": pure_prompt,
                "system_prompt": system_prompt,
            })
            original_indices.append(prompt_data["original_index"])

        print(f"Generating {len(prompts)} completions for {probe_variant}")
        outputs = llm.generate(prompts, sampling_params)
        completions_all = [[comp.text for comp in out.outputs] for out in outputs]

        variant_results = []
        for comp_list, metadata, choices, orig_idx in zip(
            completions_all, metadata_list, choices_list, original_indices
        ):
            first_completion = comp_list[0] if comp_list else ""
            if args.reasoning_generation_mode == "noncommittal":
                predicted_letter = "N/A"
                is_correct = False
                thinking_with_tags = extract_thinking_with_tags(first_completion)
            else:
                predicted_letter = extract_answer_from_completion(first_completion, exp.reasoning_mode)
                is_correct = (predicted_letter == metadata["gold"]) and (predicted_letter != "UNKNOWN")
                thinking_with_tags = None

            result_dict = {
                "qid": metadata["qid"],
                "probe_variant": probe_variant,
                "gold": metadata["gold"],
                "output_letter": predicted_letter,
                "output_correct": is_correct,
                "completions": comp_list,
                "user_prompt": metadata["user_prompt"],
                "system_prompt": metadata["system_prompt"],
                "choices": choices,
                "generation_params": {
                    "temperature": sampling_params.temperature,
                    "top_p": sampling_params.top_p,
                    "top_k": sampling_params.top_k,
                    "max_tokens": sampling_params.max_tokens,
                    "num_completions": sampling_params.n,
                },
                "reasoning_generation_mode": args.reasoning_generation_mode,
                "original_index": orig_idx,
            }
            if thinking_with_tags is not None:
                result_dict["thinking_with_tags"] = thinking_with_tags
            if args.num_completions > 1 and args.reasoning_generation_mode != "noncommittal":
                result_dict["extracted_answers"] = [
                    extract_answer_from_completion(comp, exp.reasoning_mode) for comp in comp_list
                ]
            variant_results.append(result_dict)

        # Sort by original index to maintain input order, then strip tmp field
        variant_results.sort(key=lambda x: x["original_index"])
        for r in variant_results:
            r.pop("original_index", None)

        # Write results
        single_variant_custom_output = len(probe_variants) == 1 and args.output_jsonl
        if single_variant_custom_output:
            output_path = pathlib.Path(args.output_jsonl)
        else:
            output_dir = ROOT / "results" / args.experiment_name
            output_path = output_dir / f"{probe_variant}_vllm_reasoning.jsonl"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            for r in variant_results:
                f.write(json.dumps(r) + '\n')
        print(f"✓ Wrote {len(variant_results)} results to {output_path}")

        if args.reasoning_generation_mode != "noncommittal":
            correct_count = sum(1 for r in variant_results if r["output_correct"])
            unknown_count = sum(1 for r in variant_results if r["output_letter"] == "UNKNOWN")
            accuracy = correct_count / len(variant_results) * 100 if variant_results else 0
            print(f"{probe_variant} accuracy: {accuracy:.2f}% ({correct_count}/{len(variant_results)})")
            if unknown_count > 0:
                print(f"WARNING: {unknown_count} predictions were UNKNOWN for {probe_variant}")

    print("\nvLLM generation completed successfully!")


if __name__ == "__main__":
    main()
