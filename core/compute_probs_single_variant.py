#!/usr/bin/env python3
"""
compute_probs_single_variant.py - Compute answer probabilities for a single probe variant.

This module computes probability distributions over answer choices using HuggingFace models.
For reasoning models, it supports using pre-generated reasoning from vLLM.

Key Features:
- Computes P(answer | prompt) for normal mode
- Computes P(answer | prompt + reasoning) for reasoning mode with vLLM-generated reasoning
- Supports both single and batch processing
- Extracts logits for answer tokens (A/B/C/D/E) from model outputs

Reasoning Mode Workflow:
1. First run generate_with_vllm.py to generate noncommittal reasoning
2. Then run this script with --reasoning_probing_method use_vllm_reasoning
3. This computes probabilities conditioned on the pre-generated reasoning

Usage:
    # Single GPU (baseline)
    python core/compute_probs_single_variant.py \
        --experiment_name csqa__qwen3_1_7b__d1nu1nin__nocot \
        --probe_variant bare \
        --eval_jsonl data/processed_datasets/csqa_default_split/test.jsonl \
        --output_jsonl probs.jsonl
        
    # Multi-GPU with accelerate (splits data across GPUs)
    accelerate launch --num_processes 4 core/compute_probs_single_variant.py \
        --experiment_name csqa__qwen3_1_7b__d1nu1nin__nocot \
        --probe_variant bare \
        --eval_jsonl data/processed_datasets/csqa_default_split/test.jsonl \
        --output_jsonl probs.jsonl \
        --batch_size 32  # Can use larger batch sizes with more GPUs
        
    # Reasoning mode with vLLM-generated reasoning
    # Step 1: Generate reasoning with vLLM (using defaults)
    python core/generate_with_vllm.py \
        --experiment_name csqa__qwen3_1_7br__d1nu1nin__nocot \
        --probe_variant bare \
        --eval_jsonl data/processed_datasets/csqa_default_split/test.jsonl
    # Saves to: results/csqa__qwen3_1_7br__d1nu1nin__nocot/bare_vllm_reasoning.jsonl
        
    # Step 2: Compute probabilities with the reasoning (using defaults)
    python core/compute_probs_single_variant.py \
        --experiment_name csqa__qwen3_1_7br__d1nu1nin__nocot \
        --probe_variant bare \
        --eval_jsonl data/processed_datasets/csqa_default_split/test.jsonl \
        --output_jsonl probs_with_reasoning.jsonl
    # Auto-detects reasoning mode and uses vLLM reasoning from default location
    
    # Custom batch size
    python core/compute_probs_single_variant.py \
        --experiment_name csqa__qwen3_1_7br__d1nu1nin__nocot \
        --probe_variant bare \
        --eval_jsonl data/processed_datasets/csqa_default_split/test.jsonl \
        --output_jsonl output.jsonl \
        --batch_size 32
        
    # Single processing mode (useful for debugging)
    python core/compute_probs_single_variant.py \
        --experiment_name csqa__qwen3_1_7br__d1nu1nin__nocot \
        --probe_variant bare \
        --eval_jsonl data/processed_datasets/csqa_default_split/test.jsonl \
        --output_jsonl output.jsonl \
        --single_processing
"""

import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["NCCL_P2P_DISABLE"] = "1"  # Disable P2P to avoid NCCL hang issues

import argparse
import copy
import json
import pathlib
import sys
from typing import Dict, List, Tuple, Optional, Any
import torch
import torch.nn.functional as F
from tqdm import tqdm
from datetime import datetime
from accelerate import Accelerator

# Add project root to path
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from core.exp_name import parse_experiment_name
from core.prompt_style import build_chat_prompt, load_tier_sentences, load_reasoning_file
from transformers import AutoTokenizer, AutoModelForCausalLM




def compute_answer_logprobs_simple(
    model,
    tokenizer,
    chat_prompt: str,  # Full prompt with chat template already applied
    choices: List[str],
    device: str = "cuda",
    reasoning_mode: bool = False,
    reasoning_probing_method: str = "generate_then_probe",
    **kwargs
) -> Tuple[List[float], str, float, Optional[str], Optional[Dict]]:
    """
    Compute log probabilities for answer choices with a single forward pass.
    
    For normal mode:
    1. Run model once with prompt ending in "Answer:"
    2. Look at top K tokens and decode them to find any that normalize to A/B/C/D/E
    3. For each answer letter, use the highest logit among all tokens that decode to it
    
    For reasoning mode (Qwen3 only) with reasoning_probing_method="use_vllm_reasoning":
    1. Load pre-generated reasoning from vLLM (generated with noncommittal instructions)
    2. Append the reasoning content within <think> tags
    3. Append " Answer: " after the thinking
    4. Process as normal mode to get logits
    This gives P(answer | question, noncommittal_reasoning)
    
    Returns:
        - List of log probabilities for each choice (raw logits)
        - Predicted answer letter (highest probability)
        - Highest probability value
        - Full generated text (only for reasoning mode with generate_then_probe, None otherwise)
        - Debug info (only for UNKNOWN cases, None otherwise)
    """
    # Handle reasoning mode
    full_generated_text = None
    if reasoning_mode:
        # Use pre-generated reasoning from vLLM
        if "vllm_reasoning" not in kwargs:
            raise ValueError("vllm_reasoning must be provided for reasoning mode")
        vllm_reasoning = kwargs["vllm_reasoning"]
        
        # Build full prompt: original prompt + vLLM reasoning + " Answer: "
        full_prompt = chat_prompt + vllm_reasoning + " Answer: "
            
        # Tokenize the full prompt
        inputs = tokenizer(full_prompt, return_tensors="pt", truncation=False).to(device)
    else:
        # Normal mode - tokenize the prompt as-is
        inputs = tokenizer(chat_prompt, return_tensors="pt", truncation=False).to(device)
    
    # Single forward pass
    with torch.no_grad():
        outputs = model(inputs["input_ids"])
        # Get logits for the next token position (after "Answer:")
        next_token_logits = outputs.logits[0, -1, :]  # Shape: (vocab_size,)
        
        # Get top K tokens to check for answer letters
        # We check more tokens because answers can be encoded in various ways
        top_k = 100  # Check top 100 tokens
        top_logits, top_indices = torch.topk(next_token_logits, top_k)
        
        # Map each answer letter to its highest logit among all matching tokens
        # Generate letters based on number of choices (A, B, C, ...)
        valid_letters = [chr(ord('A') + i) for i in range(len(choices))]
        letter_to_max_logit = {letter: -float('inf') for letter in valid_letters}
        
        # Check each top token to see if it decodes to an answer letter
        found_letter_tokens = []  # Debug tracking
        for rank, (logit, token_id) in enumerate(zip(top_logits, top_indices)):
            # Decode the token
            decoded = tokenizer.decode([token_id.item()], skip_special_tokens=False)
            
            # Strip whitespace and control characters
            stripped = decoded.strip()
            
            # Check if stripped result is exactly a single answer letter
            if stripped and len(stripped) == 1 and stripped.upper() in valid_letters:
                letter = stripped.upper()
                # Update max logit for this letter
                letter_to_max_logit[letter] = max(letter_to_max_logit[letter], logit.item())
                if len(found_letter_tokens) < 20:  # Track first 20 found
                    found_letter_tokens.append((rank, letter, decoded, logit.item()))
        
        
        # Build choice logits in order and check if we found any tokens
        choice_logits: List[float] = []
        found_tokens = []
        for i in range(len(choices)):
            letter = chr(ord('A') + i)
            logit = letter_to_max_logit[letter]
            if logit != -float('inf'):
                found_tokens.append(letter)
            else:
                # Use -100 for missing tokens (very negative, near-zero prob after softmax)
                logit = -100.0
            choice_logits.append(logit)

        # Check if we found any answer tokens
        if not found_tokens:
            # Suppress intermediate warnings to avoid synchronization issues
            # The final UNKNOWN count will be reported at the end
            # Set all to same value so they have equal probability
            choice_logits = [0.0] * len(choices)  # Equal logits = equal probabilities
            predicted_letter = "UNKNOWN"
            choice_probs = [1.0 / len(choices)] * len(choices)
            max_prob = 1.0 / len(choices)
            
            # Log UNKNOWN case details for debugging
            debug_info = {
                "top_tokens": [(tokenizer.decode([token_id.item()], skip_special_tokens=False), logit.item()) 
                              for logit, token_id in zip(top_logits[:20], top_indices[:20])]
            }
            return choice_logits, predicted_letter, max_prob, full_generated_text, debug_info
        else:
            # Compute probabilities over the choice logits and select the top choice
            logits_tensor = torch.tensor(choice_logits, device=next_token_logits.device)
            choice_probs = torch.softmax(logits_tensor, dim=0).tolist()
            max_idx = max(range(len(choice_probs)), key=lambda i: choice_probs[i])
            predicted_letter = chr(ord('A') + max_idx)
            max_prob = choice_probs[max_idx]
    
    return choice_logits, predicted_letter, max_prob, full_generated_text, None


def compute_answer_logprobs_batch(
    model,
    tokenizer, 
    chat_prompts: List[str],
    choices_list: List[List[str]],
    device: str = "cuda",
    batch_size: int = 8,
    reasoning_mode: bool = False,
    reasoning_probing_method: str = "generate_then_probe",
    show_progress: bool = False,
    vllm_reasoning_list: Optional[List[str]] = None,
    exp: Optional[Any] = None
) -> List[Tuple[List[float], str, float, Optional[str], Optional[Dict]]]:
    """
    True batch processing version - processes multiple prompts in a single forward pass.
    
    For normal mode:
    - Directly computes logprobs from the prompts
    
    For reasoning mode with "use_vllm_reasoning":
    - Uses pre-generated reasoning from vLLM
    - Appends reasoning + " Answer: " to prompts
    
    Returns list of (logprobs, predicted_letter, max_prob, full_generated) for each prompt.
    """
    all_results = []
    
    # Handle reasoning mode
    if reasoning_mode:
        # Use pre-generated reasoning from vLLM
        if vllm_reasoning_list is None:
            raise ValueError("vllm_reasoning_list must be provided for reasoning mode")
        if len(vllm_reasoning_list) != len(chat_prompts):
            raise ValueError(
                f"Reasoning entries mismatch: got {len(vllm_reasoning_list)}, expected {len(chat_prompts)}"
            )
        # Ensure no missing entries
        missing = [i for i, r in enumerate(vllm_reasoning_list) if r is None or r == ""]
        if missing:
            raise ValueError(
                f"Missing {len(missing)} reasoning entries in provided list; cannot proceed."
            )
        # Build new prompts with vLLM reasoning + " Answer: "
        reasoning_prompts = []
        for i, vllm_reasoning in enumerate(vllm_reasoning_list):
            new_prompt = chat_prompts[i] + vllm_reasoning + " Answer: "
            reasoning_prompts.append(new_prompt)
            
        
        prompts_to_process = reasoning_prompts
        full_generated_texts = [None] * len(chat_prompts)
    else:
        # Normal mode
        prompts_to_process = chat_prompts
        full_generated_texts = [None] * len(chat_prompts)
    
    # Process in batches for logprob computation
    result_idx = 0
    
    # Keep normal batch size for all models
    effective_batch_size = batch_size
    
    num_batches = (len(prompts_to_process) + effective_batch_size - 1) // effective_batch_size
    
    # Create iterator with optional progress bar
    if show_progress:
        batch_iterator = tqdm(range(0, len(prompts_to_process), effective_batch_size), 
                             desc="Processing batches", 
                             total=num_batches)
    else:
        batch_iterator = range(0, len(prompts_to_process), effective_batch_size)
    
    for batch_start in batch_iterator:
        batch_end = min(batch_start + effective_batch_size, len(prompts_to_process))
        batch_prompts = prompts_to_process[batch_start:batch_end]
        batch_choices = choices_list[batch_start:batch_end]
        
        # Tokenize all prompts in the batch with padding
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=False
        ).to(device)
        
        # Single forward pass for entire batch
        with torch.no_grad():
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"]
            )
            batch_logits = outputs.logits  # Shape: (batch_size, seq_len, vocab_size)
        
        # Process each item in the batch
        for i in range(len(batch_prompts)):
            # Find the last real token position (before padding)
            seq_len = inputs["attention_mask"][i].sum().item()
            next_token_logits = batch_logits[i, seq_len - 1, :]
            
            # Get top K tokens to check for answer letters
            top_k = 100  # Check top 100 tokens
            top_logits, top_indices = torch.topk(next_token_logits, top_k)
            
            # Map each answer letter to its highest logit among all matching tokens
            # Generate letters based on number of choices for this item
            valid_letters = [chr(ord('A') + j) for j in range(len(batch_choices[i]))]
            letter_to_max_logit = {letter: -float('inf') for letter in valid_letters}
            
            # Check each top token to see if it decodes to an answer letter
            for logit, token_id in zip(top_logits, top_indices):
                # Decode the token
                decoded = tokenizer.decode([token_id.item()], skip_special_tokens=False)
                
                # Strip whitespace and control characters
                stripped = decoded.strip()
                
                # Check if stripped result is exactly a single answer letter
                if stripped and len(stripped) == 1 and stripped.upper() in valid_letters:
                    letter = stripped.upper()
                    # Update max logit for this letter
                    letter_to_max_logit[letter] = max(letter_to_max_logit[letter], logit.item())
            
            # Build choice logits in order and check if we found any tokens
            choice_logits: List[float] = []
            found_tokens = []
            for j in range(len(batch_choices[i])):
                letter = chr(ord('A') + j)
                logit = letter_to_max_logit[letter]
                if logit != -float('inf'):
                    found_tokens.append(letter)
                else:
                    # Use -100 for missing tokens (very negative, near-zero prob after softmax)
                    logit = -100.0
                choice_logits.append(logit)
            
            # Check if we found any answer tokens
            if not found_tokens:
                # Suppress intermediate warnings to avoid synchronization issues
                # The final UNKNOWN count will be reported at the end
                # Set all to same value so they have equal probability
                choice_logits = [0.0] * len(batch_choices[i])  # Equal logits = equal probabilities
                predicted_letter = "UNKNOWN"
                choice_probs = [1.0 / len(batch_choices[i])] * len(batch_choices[i])
                max_prob = 1.0 / len(batch_choices[i])
                
                # Collect debug info for UNKNOWN case
                debug_info = {
                    "top_tokens": [(tokenizer.decode([token_id.item()], skip_special_tokens=False), logit.item()) 
                                  for logit, token_id in zip(top_logits[:20], top_indices[:20])]
                }
                all_results.append((choice_logits, predicted_letter, max_prob, full_generated_texts[result_idx], debug_info))
                result_idx += 1
                continue
            else:
                # Compute probabilities over the choice logits and select the top choice
                logits_tensor = torch.tensor(choice_logits, device=next_token_logits.device)
                choice_probs = torch.softmax(logits_tensor, dim=0).tolist()
                max_idx = max(range(len(choice_probs)), key=lambda i: choice_probs[i])
                predicted_letter = chr(ord('A') + max_idx)
                max_prob = choice_probs[max_idx]
            
            all_results.append((choice_logits, predicted_letter, max_prob, full_generated_texts[result_idx], None))
            result_idx += 1
    
    return all_results


def main():
    # Initialize accelerator with explicit settings to avoid NCCL issues
    accelerator = Accelerator(
        mixed_precision=None,  # Explicitly set to avoid warnings
        gradient_accumulation_steps=1,
    )
    
    parser = argparse.ArgumentParser(description="Simplified probability computation for single probe variant")
    parser.add_argument("--experiment_name", required=True)
    parser.add_argument("--probe_variant", required=True,
                       choices=["bare", "upos", "uneg", "dpos", "dneg",
                                "updp", "updn", "undp", "undn", 
                                "dpup", "dpun", "dnup", "dnun"])
    parser.add_argument("--eval_jsonl", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--canonical_wrong_jsonl", type=str,
                       help="Path to canonical wrong answers (required for non-bare variants)")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for processing")
    parser.add_argument("--single_processing", action="store_true", 
                       help="Use single processing instead of batch (for debugging)")
    parser.add_argument("--reasoning_probing_method", type=str, default=None,
                       choices=["use_vllm_reasoning"],
                       help="Method for probing in reasoning mode (default: auto-detect based on model)")
    parser.add_argument("--vllm_reasoning_jsonl", type=str,
                       help="Path to vLLM-generated reasoning (required for use_vllm_reasoning method)")
    args = parser.parse_args()
    
    # Parse experiment name
    exp = parse_experiment_name(args.experiment_name)
    
    
    # Auto-detect reasoning probing method
    if args.reasoning_probing_method is None:
        if exp.reasoning_mode:
            args.reasoning_probing_method = "use_vllm_reasoning"
        # else: stays None for non-reasoning models
    
    # Initialize vllm_reasoning_map
    vllm_reasoning_map = {}
    
    # Check if reasoning mode is supported
    if exp.reasoning_mode:
        # Only Qwen3 family supports reasoning mode
        if not exp.model_key.startswith('qwen3'):
            raise ValueError(f"Reasoning mode is only supported for Qwen3 family models, not {exp.model_key}")
        
        if args.reasoning_probing_method != "use_vllm_reasoning":
            raise ValueError("Only 'use_vllm_reasoning' method is supported for reasoning mode")

        print(f"Running in reasoning mode with '{args.reasoning_probing_method}' probing method")

        reasoning_jsonl = args.vllm_reasoning_jsonl
        file_pattern = "_vllm_reasoning.jsonl"
        backend = "vLLM"

        if not reasoning_jsonl:
            # Default path: results/{experiment_name}/{probe_variant}_vllm_reasoning.jsonl
            default_path = ROOT / "results" / args.experiment_name / f"{args.probe_variant}{file_pattern}"
            if default_path.exists():
                reasoning_jsonl = str(default_path)
                print(f"Using default {backend} reasoning file: {reasoning_jsonl}")
            else:
                raise ValueError(f"--vllm_reasoning_jsonl is required for {args.reasoning_probing_method} method. "
                               f"Default file not found at: {default_path}")

        # Use unified loader with validation
        vllm_reasoning_map = load_reasoning_file(
            reasoning_jsonl,
            eval_jsonl_path=args.eval_jsonl,
            validate_completeness=True,
            backend=backend.lower()
        )
    
    # COT is not supported
    if exp.use_cot:
        raise ValueError("COT evaluation is not supported in this simplified method")
    
    # Load model and tokenizer
    if accelerator.is_main_process:
        print(f"Loading model: {exp.hf_model_id}")
    
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
        # Special handling for gemma3 models - device_map causes issues
        # Also test special handling for qwen3 reasoning mode
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
        # Special handling for gemma3 models - device_map="auto" causes issues
        # Also test special handling for qwen3 reasoning mode
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
    
    # Load canonical wrong answers if needed
    canonical_wrong_map = {}
    if args.probe_variant != "bare":
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
    if args.probe_variant != "bare":
        tier_sentences_data = load_tier_sentences(
            exp.dataset,
            exp.model_key,
            exp.instruction,
            exp.use_cot,
            exp.reasoning_mode
        )
        if not tier_sentences_data:
            raise ValueError(f"Tier sentences not found. Run dataset_tier_generator.py first.")
    
    # Prepare prompts
    print("Preparing prompts...")
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
            gold_text = choices[row["choices"]["label"].index(row["answerKey"])]
            gold_index = choices.index(gold_text)
            gold_letter = chr(ord('A') + gold_index)
        
        # Get wrong answer
        if args.probe_variant == "bare":
            wrong_letter = ""
        else:
            wrong_letter = canonical_wrong_map.get(qid, "")
            if not wrong_letter:
                raise ValueError(f"No canonical wrong answer found for qid {qid}")
        
        # Get tier sentences
        question_tier_sentences = tier_sentences_data.get(qid) if tier_sentences_data else None
        
        # Build prompt with chat template applied
        # For use_vllm_reasoning, we need to use normal mode prompt/system instruction
        use_vllm_reasoning = exp.reasoning_mode and args.reasoning_probing_method == "use_vllm_reasoning"
        
        chat_prompt, pure_prompt, system_prompt = build_chat_prompt(
            tokenizer=tokenizer,
            question=question,
            answer_choices=choices,
            exp=exp,
            probe_variant=args.probe_variant,
            gold_answer=gold_letter,
            wrong_answer=wrong_letter,
            tier_sentences=question_tier_sentences,
            force_answer_prompt=True,  # Always add Answer: for HF probing
            use_vllm_reasoning=use_vllm_reasoning  # Pass flag to get normal system prompt
        )
        
        # Store the chat prompt
        prompts.append(chat_prompt)
        choices_list.append(choices)
        metadata_list.append({
            "qid": qid,
            "gold": gold_letter,
            "user_prompt": pure_prompt,
            "system_prompt": system_prompt
        })
    
    # Split data across processes for multi-GPU
    if accelerator.num_processes > 1:
        # Calculate splits for each process
        total_samples = len(prompts)
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
        prompts = prompts[start_idx:end_idx]
        choices_list = choices_list[start_idx:end_idx]
        metadata_list = metadata_list[start_idx:end_idx]
        eval_data = eval_data[start_idx:end_idx]
        
        print(f"Process {accelerator.process_index}: Processing samples {start_idx} to {end_idx} ({len(prompts)} total)")
    
    # Prepare UNKNOWN log file if needed (only on main process)
    unknown_log_path = None
    if args.output_jsonl and accelerator.is_main_process:
        output_dir = pathlib.Path(args.output_jsonl).parent
        unknown_log_path = output_dir / f"unknown_cases_{args.experiment_name}_{args.probe_variant}.jsonl"
    
    # Compute probabilities
    if accelerator.is_main_process:
        print(f"Computing probabilities for {len(prompts)} prompts per process...")
    
    unknown_cases = []
    
    if args.single_processing:
        # Single processing mode
        results = []
        for idx, (prompt, choices) in enumerate(tqdm(zip(prompts, choices_list), 
                                   total=len(prompts), desc="Processing",
                                   disable=not accelerator.is_main_process)):
            # Get vLLM reasoning if using that method
            kwargs = {}
            if exp.reasoning_mode and args.reasoning_probing_method == "use_vllm_reasoning":
                qid = metadata_list[idx]["qid"]
                kwargs["vllm_reasoning"] = vllm_reasoning_map.get(qid, "")
            
            result = compute_answer_logprobs_simple(
                model, tokenizer, prompt, choices, device, 
                reasoning_mode=exp.reasoning_mode,
                reasoning_probing_method=args.reasoning_probing_method,
                **kwargs
            )
            # Unpack with optional debug info
            logprobs, predicted_letter, max_prob, full_generated, debug_info = result
            if debug_info and predicted_letter == "UNKNOWN":
                # Simplified unknown case tracking to avoid gather issues
                unknown_cases.append({
                    "qid": metadata_list[idx]["qid"]
                })
            results.append((logprobs, predicted_letter, max_prob, full_generated))
    else:
        # Batch processing mode
        results = []
        for i in tqdm(range(0, len(prompts), args.batch_size), desc="Processing batches",
                     disable=not accelerator.is_main_process):
            batch_prompts = prompts[i:i + args.batch_size]
            batch_choices = choices_list[i:i + args.batch_size]
            
            # Get vLLM reasoning for batch if using that method
            vllm_reasoning_list = None
            if exp.reasoning_mode and args.reasoning_probing_method == "use_vllm_reasoning":
                vllm_reasoning_list = []
                for j in range(len(batch_prompts)):
                    if i + j < len(metadata_list):
                        qid = metadata_list[i + j]["qid"]
                        vllm_reasoning_list.append(vllm_reasoning_map.get(qid, ""))
            
            batch_results = compute_answer_logprobs_batch(
                model, tokenizer, batch_prompts, batch_choices, device, args.batch_size,
                reasoning_mode=exp.reasoning_mode,
                reasoning_probing_method=args.reasoning_probing_method,
                vllm_reasoning_list=vllm_reasoning_list,
                exp=exp
            )
            # Process results and collect UNKNOWN cases
            for j, result in enumerate(batch_results):
                batch_idx = i + j
                if len(result) == 5 and result[4] is not None and result[1] == "UNKNOWN":
                    # Simplified unknown case tracking to avoid gather issues
                    unknown_cases.append({
                        "qid": metadata_list[batch_idx]["qid"]
                    })
                results.append(result[:4])  # Append without debug info
    
    # Gather results from all processes
    if accelerator.num_processes > 1:
        # Gather all results, metadata, and choices
        all_results = accelerator.gather_for_metrics(results)
        all_metadata = accelerator.gather_for_metrics(metadata_list)
        all_choices = accelerator.gather_for_metrics(choices_list)
        # Skip gathering unknown_cases - it's not used and causes hanging
        # all_unknown_cases = accelerator.gather_for_metrics(unknown_cases)
        
        # Flatten lists if gathered
        if accelerator.is_main_process:
            results = all_results
            metadata_list = all_metadata
            choices_list = all_choices
            # unknown_cases = all_unknown_cases
    
    # Save results (only on main process)
    if accelerator.is_main_process:
        output_path = pathlib.Path(args.output_jsonl)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            for (logprobs, predicted_letter, max_prob, full_generated), metadata, choices in zip(results, metadata_list, choices_list):
                # Convert raw logits to probabilities using softmax over choices
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
                    "probe_variant": args.probe_variant,
                    "gold": metadata["gold"],
                    "output_letter": predicted_letter,
                    "output_correct": output_correct,
                    "probs": letter_probs,
                    "user_prompt": metadata["user_prompt"],
                    "system_prompt": metadata["system_prompt"],
                    "max_prob": max_prob,
                    "choices": choices
                }
                
                # Add reasoning mode information
                if exp.reasoning_mode:
                    result_dict["reasoning_probing_method"] = args.reasoning_probing_method
                
                # Add full generated text for reasoning mode with generate_then_probe
                if full_generated is not None:
                    result_dict["reasoning_mode_generated_text"] = full_generated
                
                f.write(json.dumps(result_dict) + '\n')
    
        print(f"✓ Wrote {len(results)} results to {output_path}")
        
        # Print accuracy and UNKNOWN count
        correct = sum(1 for r, m in zip(results, metadata_list) if r[1] == m["gold"] and r[1] != "UNKNOWN")
        unknown_count = sum(1 for r in results if r[1] == "UNKNOWN")
        accuracy = correct / len(results) * 100
        print(f"Accuracy: {accuracy:.2f}% ({correct}/{len(results)})")
        # Suppress UNKNOWN warning to avoid hanging issues
        # if unknown_count > 0:
        #     print(f"WARNING: {unknown_count} predictions were UNKNOWN (no answer tokens found in top 100)")
            
            # Write UNKNOWN cases log - DISABLED to avoid potential file system hang
            # if unknown_cases and unknown_log_path:
            #     with open(unknown_log_path, 'w') as f:
            #         for case in unknown_cases:
            #             f.write(json.dumps(case) + '\n')
            #     print(f"✓ Logged {len(unknown_cases)} UNKNOWN cases to {unknown_log_path}")
        
        # Flush stdout to ensure all output is written before synchronization
        sys.stdout.flush()
    
    # Ensure all processes complete before exiting
    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
