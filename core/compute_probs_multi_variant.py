#!/usr/bin/env python3
"""
compute_probs_multi_variant.py - Batch processing for multiple probe variants with priors.

This module processes multiple probe variants that include prior statements (user/doc assertions)
with a single model load for efficiency.

Reasoning Mode Support:
For reasoning models (Qwen3 family), supports using pre-generated reasoning from vLLM.
First run generate_with_vllm.py to generate noncommittal reasoning, then use this script
with --vllm_reasoning_jsonl to compute probabilities conditioned on the pre-generated reasoning.

Usage:
    # Process multiple probe variants at once (excluding 'bare')
    python core/compute_probs_multi_variant.py \
        --experiment_name csqa__llama3_2_1b_instruct__d1nu1nin__nocot \
        --probe_variants upos,uneg,dpos \
        --eval_jsonl data/processed_datasets/csqa_default_split/test.jsonl \
        --output_dir results/csqa__llama3_2_1b_instruct__d1nu1nin__nocot \
        --batch_size 16
        
    # Multi-GPU with accelerate (distributes probe variants across GPUs)
    accelerate launch --num_processes 4 core/compute_probs_multi_variant.py \
        --experiment_name csqa__llama3_2_1b_instruct__d1nu1nin__nocot \
        --probe_variants upos,uneg,dpos,dneg,updp,updn,undp,undn \
        --eval_jsonl data/processed_datasets/csqa_default_split/test.jsonl \
        --output_dir results/csqa__llama3_2_1b_instruct__d1nu1nin__nocot \
        --batch_size 32
        
    # For reasoning models with vLLM-generated reasoning
    # Step 1: Generate reasoning with vLLM for all variants at once (efficient)
    python core/generate_with_vllm.py \
        --experiment_name csqa__qwen3_1_7br__d1nu1nin__nocot \
        --probe_variants upos,uneg,dpos,dneg \
        --eval_jsonl data/processed_datasets/csqa_default_split/test.jsonl
    # Outputs: results/csqa__qwen3_1_7br__d1nu1nin__nocot/{variant}_vllm_reasoning.jsonl
        
    # Step 2: Compute probabilities with the reasoning (using defaults)
    python core/compute_probs_multi_variant.py \
        --experiment_name csqa__qwen3_1_7br__d1nu1nin__nocot \
        --probe_variants upos,uneg,dpos,dneg \
        --eval_jsonl data/processed_datasets/csqa_default_split/test.jsonl \
        --output_dir results/csqa__qwen3_1_7br__d1nu1nin__nocot
    # Auto-detects reasoning files: {probe_variant}_vllm_reasoning.jsonl for each variant
        
    # Full workflow for reasoning model with all variants
    # Step 1: Generate reasoning for bare variant
    python core/generate_with_vllm.py \
        --experiment_name csqa__qwen3_1_7br__d1nu1nin__nocot \
        --probe_variant bare \
        --eval_jsonl data/processed_datasets/csqa_default_split/test.jsonl
        
    # Step 2: Process bare variant with HF probing
    python core/compute_probs_single_variant.py \
        --experiment_name csqa__qwen3_1_7br__d1nu1nin__nocot \
        --probe_variant bare \
        --eval_jsonl data/processed_datasets/csqa_default_split/test.jsonl \
        --output_jsonl results/csqa__qwen3_1_7br__d1nu1nin__nocot/probs_bare.jsonl
    # Auto-detects reasoning file: bare_vllm_reasoning.jsonl
        
    # Step 3: Generate reasoning for remaining 8 variants efficiently
    python core/generate_with_vllm.py \
        --experiment_name csqa__qwen3_1_7br__d1nu1nin__nocot \
        --probe_variants upos,uneg,dpos,dneg,updp,updn,undp,undn \
        --eval_jsonl data/processed_datasets/csqa_default_split/test.jsonl \
        --canonical_wrong_jsonl results/csqa__qwen3_1_7br__d1nu1nin__nocot/canonical_wrong.jsonl
        
    # Step 4: Process all non-bare variants together with HF probing
    python core/compute_probs_multi_variant.py \
        --experiment_name csqa__qwen3_1_7br__d1nu1nin__nocot \
        --probe_variants upos,uneg,dpos,dneg,updp,updn,undp,undn \
        --eval_jsonl data/processed_datasets/csqa_default_split/test.jsonl \
        --output_dir results/csqa__qwen3_1_7br__d1nu1nin__nocot
    # Auto-detects reasoning files for each variant
    
    # Or specify a single reasoning file for all variants (not recommended)
    python core/compute_probs_multi_variant.py \
        --experiment_name csqa__qwen3_1_7br__d1nu1nin__nocot \
        --probe_variants upos,uneg,dpos,dneg \
        --eval_jsonl data/processed_datasets/csqa_default_split/test.jsonl \
        --output_dir results/csqa__qwen3_1_7br__d1nu1nin__nocot \
        --vllm_reasoning_jsonl results/csqa__qwen3_1_7br__d1nu1nin__nocot/bare_vllm_reasoning.jsonl
        
    # Note: The 'bare' variant must be processed separately with compute_probs_single_variant.py:
    python core/compute_probs_single_variant.py \
        --experiment_name csqa__qwen3_1_7b__d1nu1nin__nocot \
        --probe_variant bare \
        --eval_jsonl data/processed_datasets/csqa_default_split/test.jsonl \
        --output_jsonl results/csqa__qwen3_1_7b__d1nu1nin__nocot/probs_bare.jsonl
"""

import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["NCCL_P2P_DISABLE"] = "1"  # Disable P2P to avoid NCCL hang issues

import argparse
import gc
import json
import pathlib
import sys
from typing import Dict, List, Tuple, Optional
import torch
import torch.nn.functional as F
from tqdm import tqdm
from accelerate import Accelerator

# Add project root to path
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from core.exp_name import parse_experiment_name
from core.prompt_style import build_chat_prompt, load_tier_sentences, load_reasoning_file
from transformers import AutoTokenizer, AutoModelForCausalLM

# All available probe variants (excluding "bare" which has no priors)
# Note: "bare" should be processed separately with compute_probs_single_variant.py
ALL_PROBE_VARIANTS = [
    "upos", "uneg", "dpos", "dneg",  # Single prior variants
    "updp", "updn", "undp", "undn",  # Double prior: user first
    "dpup", "dpun", "dnup", "dnun"   # Double prior: doc first
]


def compute_answer_logprobs_for_variant(
    model,
    tokenizer,
    chat_prompts: List[str],
    choices_list: List[List[str]],
    device: str = "cuda",
    batch_size: int = 8,
    reasoning_mode: bool = False,
    reasoning_probing_method: str = "generate_then_probe",
    vllm_reasoning_list: Optional[List[str]] = None
) -> List[Tuple[List[float], str, float, Optional[str], Optional[Dict]]]:
    """
    Compute answer logprobs for a list of prompts (single variant).
    
    Simply calls the implementation from compute_probs_single_variant.
    """
    # Import the function from single_variant module
    from core.compute_probs_single_variant import compute_answer_logprobs_batch
    
    return compute_answer_logprobs_batch(
        model, tokenizer, chat_prompts, 
        choices_list, device, batch_size,
        reasoning_mode=reasoning_mode, 
        reasoning_probing_method=reasoning_probing_method,
        show_progress=True,  # Show progress for each variant
        vllm_reasoning_list=vllm_reasoning_list
    )


def process_probe_variant(
    model,
    tokenizer,
    exp,
    probe_variant: str,
    eval_data: List[Dict],
    canonical_wrong_map: Dict[str, str],
    tier_sentences_data: Optional[Dict],
    device: str,
    batch_size: int = 8,
    reasoning_mode: bool = False,
    reasoning_probing_method: str = "generate_then_probe",
    vllm_reasoning_map: Optional[Dict[str, str]] = None
) -> List[Dict]:
    """
    Process a single probe variant and return results.
    """
    print(f"\n{'='*40}")
    print(f"Processing probe variant: {probe_variant}")
    print(f"{'='*40}")
    
    # Prepare prompts for this variant
    prompts = []
    choices_list = []
    metadata_list = []
    
    for row in eval_data:
        qid = row["id"]
        question = row["question"]
        
        # Handle different choice formats
        if isinstance(row["choices"], dict) and "text" in row["choices"]:
            choices = row["choices"]["text"]
        else:
            choices = row["choices"]
            
        # Get gold answer
        if "answerKey" in row:
            gold_letter = row["answerKey"]
        else:
            # Find gold letter from choices
            if "label" in row["choices"]:
                gold_text = choices[row["choices"]["label"].index(row["answerKey"])]
                gold_index = choices.index(gold_text)
                gold_letter = chr(ord('A') + gold_index)
            else:
                gold_letter = row["answerKey"]
        
        # Get wrong answer
        if probe_variant == "bare":
            wrong_letter = ""
        else:
            wrong_letter = canonical_wrong_map.get(qid, "")
            if not wrong_letter:
                raise ValueError(f"No canonical wrong answer found for qid {qid}")
        
        # Get tier sentences
        question_tier_sentences = None
        if probe_variant != "bare" and tier_sentences_data:
            question_tier_sentences = tier_sentences_data.get(qid)
            if not question_tier_sentences:
                raise ValueError(f"No tier sentences found for question {qid}")
        
        # Build prompt (with chat template applied)
        # For use_vllm_reasoning, we need to use normal mode prompt/system instruction
        use_vllm_reasoning = reasoning_mode and reasoning_probing_method == "use_vllm_reasoning"
        
        chat_prompt, pure_prompt, system_prompt = build_chat_prompt(
            tokenizer=tokenizer,
            question=question,
            answer_choices=choices,
            exp=exp,
            probe_variant=probe_variant,
            gold_answer=gold_letter,
            wrong_answer=wrong_letter,
            tier_sentences=question_tier_sentences,
            force_answer_prompt=True,  # Always add Answer: for HF probing
            use_vllm_reasoning=use_vllm_reasoning  # Pass flag to get normal system prompt
        )
        
        # Store the chat prompt (with template applied)
        prompts.append(chat_prompt)
        choices_list.append(choices)
        metadata_list.append({
            "qid": qid,
            "gold": gold_letter,
            "user_prompt": pure_prompt,
            "system_prompt": system_prompt
        })
    
    # Prepare vLLM reasoning list if needed
    vllm_reasoning_list = None
    if reasoning_mode and reasoning_probing_method == "use_vllm_reasoning":
        if not vllm_reasoning_map or probe_variant not in vllm_reasoning_map:
            raise ValueError(f"vllm_reasoning_map must be provided for variant '{probe_variant}'")
        vllm_reasoning_list = []
        variant_reasoning = vllm_reasoning_map[probe_variant]
        missing_qids = []
        for metadata in metadata_list:
            qid = metadata["qid"]
            if qid not in variant_reasoning:
                missing_qids.append(qid)
            else:
                vllm_reasoning_list.append(variant_reasoning[qid])
        if missing_qids:
            raise ValueError(
                f"Missing {len(missing_qids)} reasoning entries for variant '{probe_variant}'. "
                f"Reasoning file appears incomplete. Regenerate reasoning (e.g., via run_batch_probes_efficient) "
                f"to produce {len(metadata_list)} entries matching eval set."
            )
    
    # Compute probabilities for this variant using true batch processing
    results = compute_answer_logprobs_for_variant(
        model, tokenizer, prompts, 
        choices_list, device, batch_size,
        reasoning_mode=reasoning_mode,
        reasoning_probing_method=reasoning_probing_method,
        vllm_reasoning_list=vllm_reasoning_list
    )
    
    # Format results
    formatted_results = []
    for result, metadata, choices in zip(results, metadata_list, choices_list):
        # Unpack result (now returns 5 values with optional debug info)
        if len(result) == 5:
            logprobs, predicted_letter, max_logprob, full_generated, debug_info = result
        else:
            # Backward compatibility if function returns old format
            logprobs, predicted_letter, max_logprob, full_generated = result
            debug_info = None
        # Convert to probabilities
        probs = torch.softmax(torch.tensor(logprobs), dim=0).tolist()
        
        # Create letter-based probability results
        letter_probs = []
        for j, (choice, prob, logit) in enumerate(zip(choices, probs, logprobs)):
            letter = chr(ord('A') + j)
            letter_probs.append({
                "letter": letter,
                "text": choice,
                "prob": prob,
                "logit": logit  # Raw logit from the model
            })
        
        # Determine if model is correct (UNKNOWN is always incorrect)
        output_correct = (predicted_letter == metadata["gold"]) and (predicted_letter != "UNKNOWN")
        
        result_dict = {
            "qid": metadata["qid"],
            "probe_variant": probe_variant,
            "gold": metadata["gold"],
            "output_letter": predicted_letter,
            "output_correct": output_correct,
            "probs": letter_probs,
            "user_prompt": metadata["user_prompt"],
            "system_prompt": metadata["system_prompt"],
            "max_prob": max_logprob,  # Note: This is actually max probability, not logprob
            "choices": choices
        }
        
        # Add reasoning mode information
        if reasoning_mode:
            result_dict["reasoning_probing_method"] = reasoning_probing_method
            
        # Add full generated text for reasoning mode with generate_then_probe
        if full_generated is not None:
            result_dict["reasoning_mode_generated_text"] = full_generated
        
        formatted_results.append(result_dict)
    
    # Print accuracy for this variant
    correct = sum(1 for r in formatted_results if r["output_correct"])
    unknown_count = sum(1 for r in formatted_results if r["output_letter"] == "UNKNOWN")
    accuracy = correct / len(formatted_results) * 100
    print(f"{probe_variant} accuracy: {accuracy:.2f}% ({correct}/{len(formatted_results)})")
    # Suppress UNKNOWN warning to avoid hanging issues
    # if unknown_count > 0:
    #     print(f"WARNING: {unknown_count} predictions were UNKNOWN for {probe_variant}")
    
    return formatted_results


def main():
    # Initialize accelerator with explicit settings to avoid NCCL issues
    accelerator = Accelerator(
        mixed_precision=None,  # Explicitly set to avoid warnings
        gradient_accumulation_steps=1,
    )
    
    parser = argparse.ArgumentParser(description="Batch processing for multiple probe variants")
    parser.add_argument("--experiment_name", required=True)
    parser.add_argument("--probe_variants", required=True,
                       help="Comma-separated list of probe variants or 'all'")
    parser.add_argument("--eval_jsonl", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--canonical_wrong_jsonl", type=str,
                       help="Path to canonical wrong answers (required for non-bare variants)")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for processing")
    parser.add_argument("--overwrite", action="store_true", 
                       help="Overwrite existing output files")
    parser.add_argument("--reasoning_probing_method", type=str, default=None,
                       choices=["use_vllm_reasoning"],
                       help="Method for probing in reasoning mode (default: auto-detect based on model)")
    parser.add_argument("--vllm_reasoning_jsonl", type=str,
                       help="Path to vLLM-generated reasoning (required for use_vllm_reasoning method)")
    args = parser.parse_args()
    
    # Parse probe variants
    probe_variants = [pv.strip() for pv in args.probe_variants.split(",")]
    
    # Validate probe variants
    for pv in probe_variants:
        if pv == "bare":
            raise ValueError(
                "The 'bare' probe variant cannot be used with compute_probs_multi_variant.py\n"
                "Reason: 'bare' has no prior statements, so there's no benefit to batch processing.\n"
                "Please use compute_probs_single_variant.py for the 'bare' variant."
            )
        if pv not in ALL_PROBE_VARIANTS:
            raise ValueError(f"Invalid probe variant: {pv}. Valid options: {', '.join(ALL_PROBE_VARIANTS)}")
    
    print(f"Will process probe variants: {', '.join(probe_variants)}")
    
    # Parse experiment name
    exp = parse_experiment_name(args.experiment_name)
    
    # COT is not supported
    if exp.use_cot:
        raise ValueError("COT evaluation is not supported in this simplified method")
    
    # Auto-detect reasoning probing method
    if args.reasoning_probing_method is None:
        if exp.reasoning_mode:
            args.reasoning_probing_method = "use_vllm_reasoning"
        # else: stays None for non-reasoning models
    
    # Check if reasoning mode is supported
    vllm_reasoning_map = {}
    if exp.reasoning_mode:
        # Only Qwen3 family supports reasoning mode
        if not exp.model_key.startswith('qwen3'):
            raise ValueError(f"Reasoning mode is only supported for Qwen3 family models, not {exp.model_key}")
        
        if args.reasoning_probing_method != "use_vllm_reasoning":
            raise ValueError("Only 'use_vllm_reasoning' method is supported for reasoning mode")

        print(f"Running in reasoning mode with '{args.reasoning_probing_method}' probing method")

        # Load reasoning for each probe variant
        # In multi-variant mode, we need reasoning for each variant separately
        # Structure: vllm_reasoning_map[probe_variant][qid] = reasoning
        vllm_reasoning_map = {}

        reasoning_jsonl_arg = args.vllm_reasoning_jsonl
        file_pattern = "_vllm_reasoning.jsonl"
        backend = "vLLM"
        generate_script = "generate_with_vllm.py"
        
        for probe_variant in probe_variants:
            # Check for variant-specific reasoning file
            if reasoning_jsonl_arg:
                # If a single file is provided, use it for all variants
                reasoning_file = reasoning_jsonl_arg
            else:
                # Default path: results/{experiment_name}/{probe_variant}_{backend}_reasoning.jsonl
                default_path = ROOT / "results" / args.experiment_name / f"{probe_variant}{file_pattern}"
                if default_path.exists():
                    reasoning_file = str(default_path)
                    print(f"Using default {backend} reasoning file for {probe_variant}: {reasoning_file}")
                else:
                    raise ValueError(f"{backend} reasoning file not found for variant '{probe_variant}'. "
                                   f"Expected at: {default_path}. "
                                   f"Please run {generate_script} for this variant first.")
            
            # Use unified loader with validation
            vllm_reasoning_map[probe_variant] = load_reasoning_file(
                reasoning_file,
                eval_jsonl_path=args.eval_jsonl,
                validate_completeness=True,
                backend=backend.lower()
            )
    
    # Create output directory
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check which files already exist
    if not args.overwrite:
        existing_files = []
        for pv in probe_variants:
            output_file = output_dir / f"probs_{pv}.jsonl"
            if output_file.exists():
                existing_files.append(pv)
        
        if existing_files:
            print(f"\nFound existing files for: {', '.join(existing_files)}")
            probe_variants = [pv for pv in probe_variants if pv not in existing_files]
            if not probe_variants:
                print("All requested variants already processed. Use --overwrite to reprocess.")
                return
            print(f"Will process remaining variants: {', '.join(probe_variants)}")
    
    # Load model and tokenizer ONCE
    if accelerator.is_main_process:
        print(f"\nLoading model: {exp.hf_model_id}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(exp.hf_model_id, trust_remote_code=True)
    
    # Set padding token if not already set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # CRITICAL: For gemma models in batch probing, use RIGHT padding
    # Despite documentation suggesting left padding for generation, 
    # gemma3 produces correct letter tokens with right padding in batch inference
    if exp.model_key.startswith('gemma'):
        tokenizer.padding_side = 'right'
    
    # For multi-GPU with accelerate, we need a different approach
    if accelerator.num_processes > 1:
        # Each process loads the model independently to avoid NCCL conflicts
        # Special handling for gemma3 and qwen3 reasoning models - device_map causes issues
        if exp.model_key.startswith('gemma3') or (exp.model_key.startswith('qwen3') and exp.reasoning_mode):
            model = AutoModelForCausalLM.from_pretrained(
                exp.hf_model_id,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            model = model.to(accelerator.device)
        else:
            model = AutoModelForCausalLM.from_pretrained(
                exp.hf_model_id,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                device_map={"": accelerator.device},  # Load directly to each GPU
                low_cpu_mem_usage=True
            )
    else:
        # Single GPU - use standard loading
        # Special handling for gemma3 and qwen3 reasoning models - device_map="auto" causes issues
        if exp.model_key.startswith('gemma3') or (exp.model_key.startswith('qwen3') and exp.reasoning_mode):
            model = AutoModelForCausalLM.from_pretrained(
                exp.hf_model_id,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            model = model.to(accelerator.device)
        else:
            model = AutoModelForCausalLM.from_pretrained(
                exp.hf_model_id,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                device_map="auto",
                low_cpu_mem_usage=True
            )
    
    model.eval()
    
    # Get device from accelerator
    device = accelerator.device
    print(f"Using device: {device} (Process {accelerator.process_index}/{accelerator.num_processes})")
    
    # Load evaluation data
    eval_data = []
    with open(args.eval_jsonl) as f:
        for line in f:
            eval_data.append(json.loads(line))
    print(f"Loaded {len(eval_data)} evaluation examples")
    
    # Check if we need canonical wrong answers
    non_bare_variants = [pv for pv in probe_variants if pv != "bare"]
    
    # Load canonical wrong answers if needed
    canonical_wrong_map = {}
    if non_bare_variants:
        if not args.canonical_wrong_jsonl:
            # Try default location
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
    
    # Load tier sentences if needed
    tier_sentences_data = None
    if non_bare_variants:
        tier_sentences_data = load_tier_sentences(
            exp.dataset,
            exp.model_key,
            exp.instruction,
            exp.use_cot,
            exp.reasoning_mode
        )
        if not tier_sentences_data:
            raise ValueError(f"Tier sentences not found. Run dataset_tier_generator.py first.")
        print(f"Loaded tier sentences for {len(tier_sentences_data)} questions")
    
    # Split evaluation data across processes for multi-GPU (data parallelism)
    if accelerator.num_processes > 1:
        # Each process handles different prompts, but all variants
        total_samples = len(eval_data)
        samples_per_process = total_samples // accelerator.num_processes
        remainder = total_samples % accelerator.num_processes
        
        # Calculate start and end indices for this process
        if accelerator.process_index < remainder:
            start_idx = accelerator.process_index * (samples_per_process + 1)
            end_idx = start_idx + samples_per_process + 1
        else:
            start_idx = accelerator.process_index * samples_per_process + remainder
            end_idx = start_idx + samples_per_process
        
        # Slice data for this process
        process_eval_data = eval_data[start_idx:end_idx]
        print(f"Process {accelerator.process_index}: Processing samples {start_idx} to {end_idx} ({len(process_eval_data)} total)")
    else:
        process_eval_data = eval_data
    
    # All processes handle all probe variants (but different data)
    process_variants = probe_variants
    
    # Process each probe variant
    if accelerator.is_main_process:
        print(f"\n{'='*60}")
        print(f"Processing {len(probe_variants)} probe variants total")
        print(f"{'='*60}")
    
    # Process each probe variant and save immediately to avoid OOM
    for probe_variant in probe_variants:
        # Only process variants assigned to this process
        if probe_variant in process_variants:
            # Process this variant with this process's data subset
            results = process_probe_variant(
                model=model,
                tokenizer=tokenizer,
                exp=exp,
                probe_variant=probe_variant,
                eval_data=process_eval_data,  # Use process-specific data
                canonical_wrong_map=canonical_wrong_map,
                tier_sentences_data=tier_sentences_data,
                device=device,
                batch_size=args.batch_size,
                reasoning_mode=exp.reasoning_mode,
                reasoning_probing_method=args.reasoning_probing_method,
                vllm_reasoning_map=vllm_reasoning_map
            )
        else:
            # This process doesn't handle this variant
            results = []
        
        # Gather results from all processes immediately for this variant
        if accelerator.num_processes > 1:
            gathered_results = accelerator.gather_for_metrics(results)
        else:
            gathered_results = results
        
        # Save results immediately (only main process)
        if accelerator.is_main_process:
            output_file = output_dir / f"probs_{probe_variant}.jsonl"
            
            with open(output_file, 'w') as f:
                for result in gathered_results:
                    f.write(json.dumps(result) + '\n')
            
            print(f"✓ Wrote {len(gathered_results)} results to {output_file}")
        
        # Clear results to free memory before processing next variant
        del results
        if 'gathered_results' in locals():
            del gathered_results
        
        # Force garbage collection to free memory
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Synchronize all processes before moving to next variant
        accelerator.wait_for_everyone()
    
    # Print completion message (only main process)
    if accelerator.is_main_process:
        print(f"\n{'='*60}")
        print(f"Completed processing {len(probe_variants)} probe variants")
        print(f"Output directory: {output_dir}")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()