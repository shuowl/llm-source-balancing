#!/usr/bin/env python3
"""
compute_answers_and_probs_openai.py - Generate answers and compute probability distributions using OpenAI API.

Prerequisites:
    1. Set your OpenAI API key:
       export OPENAI_API_KEY="your-key-here"
    
    2. Ensure the model is configured in configs/models_config.yaml:
       gpt_4o: "gpt-4o-2024-08-06"
       gpt_4o_mini: "gpt-4o-mini-2024-07-18"

Usage Examples:

    # 1. Basic usage with GPT-4o on CSQA dataset
    python core/compute_answers_and_probs_openai.py \
        --experiment_name csqa__gpt_4o_mini__d1nu1nin__nocot \
        --probe_variant bare \
        --eval_jsonl data/processed_datasets/csqa_default_split/test.jsonl

    # 2. GPT-4o-mini with custom output location
    python core/compute_answers_and_probs_openai.py \
        --experiment_name csqa__gpt_4o_mini__d1nu1nin__nocot \
        --probe_variant bare \
        --eval_jsonl data/processed_datasets/csqa_default_split/test.jsonl \
        --output_jsonl results/gpt4o_mini_probs.jsonl

    # 3. Running with different probe variants (requires canonical_wrong.jsonl and tier sentences)
    # First run bare probe to generate canonical_wrong.jsonl:
    python core/compute_answers_and_probs_openai.py \
        --experiment_name csqa__gpt_4o__d1nu1nin__nocot \
        --probe_variant bare \
        --eval_jsonl data/processed_datasets/csqa_default_split/test.jsonl
    
    # Then run other probe variants:
    for variant in upos dpos uneg dneg; do
        python core/compute_answers_and_probs_openai.py \
            --experiment_name csqa__gpt_4o__d1nu1nin__nocot \
            --probe_variant $variant \
            --eval_jsonl data/processed_datasets/csqa_default_split/test.jsonl
    done

    # 4. Math dataset (GSM8K) with temperature 0 for deterministic results
    python core/compute_answers_and_probs_openai.py \
        --experiment_name gsm8k__gpt_4o__d1nu1nin__nocot \
        --probe_variant bare \
        --eval_jsonl data/processed_datasets/gsm8k_default_split/test.jsonl \
        --temperature 0.0 \
        --max-concurrent 5

    # 5. Chain-of-thought mode
    python core/compute_answers_and_probs_openai.py \
        --experiment_name csqa__gpt_4o__d1nu1nin__cot \
        --probe_variant bare \
        --eval_jsonl data/processed_datasets/csqa_default_split/test.jsonl \
        --max-concurrent 5

    # 6. Debug mode to see detailed logprobs
    python core/compute_answers_and_probs_openai.py \
        --experiment_name csqa__gpt_4o_mini__d1nu1nin__nocot \
        --probe_variant bare \
        --eval_jsonl data/processed_datasets/csqa_default_split/test.jsonl \
        --debug \
        --max-concurrent 3

    # 7. Custom API key (instead of environment variable)
    python core/compute_answers_and_probs_openai.py \
        --experiment_name csqa__gpt_4o__d1nu1nin__nocot \
        --probe_variant bare \
        --eval_jsonl data/processed_datasets/csqa_default_split/test.jsonl \
        --api_key "sk-your-api-key-here"

    # 8. Larger batch size for faster processing (adjust based on rate limits)
    python core/compute_answers_and_probs_openai.py \
        --experiment_name csqa__gpt_4o_mini__d1nu1nin__nocot \
        --probe_variant bare \
        --eval_jsonl data/processed_datasets/csqa_default_split/test.jsonl \
        --max-concurrent 50

Output Format:
    The script generates a JSONL file with one line per question containing:
    {
        "qid": "question_id",
        "gold": "A",
        "predicted": "B",
        "parsed_predicted": "B",
        "raw_predicted_token": " B",
        "letter_probs": [
            {"letter": "A", "text": "choice text", "prob": 0.2, "logp": -1.609},
            {"letter": "B", "text": "choice text", "prob": 0.8, "logp": -0.223},
            ...
        ],
        "generated_text": "The full generated response",
        "probe_variant": "bare",
        "model": "gpt-4o-2024-08-06"
    }
"""

import os
import argparse
import json
import pathlib
import sys
from typing import Dict, List, Tuple, Optional
import asyncio
from dataclasses import dataclass
import re
import time
from tqdm import tqdm
import yaml

# OpenAI imports
from openai import AsyncOpenAI
import tiktoken

# Add project root to path
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from core.exp_name import parse_experiment_name
from core.prompt_style_openai import build_openai_messages, load_tier_sentences

# Fixed max concurrent requests - override any passed value
# This helps avoid rate limit issues and provides consistent behavior
FIXED_MAX_CONCURRENT = 100


@dataclass
class OpenAIGenerationConfig:
    """Configuration for OpenAI generation."""
    suffix: str  # The suffix to append to prompt
    temperature: float = 0.7
    top_p: float = 0.8
    max_tokens: int = 2048
    logprobs: bool = True
    top_logprobs: int = 20  # Get top 20 logprobs for each token
    

# Predefined configurations for different experiment types
OPENAI_GENERATION_CONFIGS = {
    # Normal mode - immediate answer expected
    "normal": OpenAIGenerationConfig(
        suffix="Answer:",
        temperature=0.7,
        top_p=0.8,
        max_tokens=5,  # We expect immediate answer
        logprobs=True,
        top_logprobs=20
    ),
    
    # CoT mode
    "cot": OpenAIGenerationConfig(
        suffix="<answer>",
        temperature=0.7,
        top_p=0.8,
        max_tokens=2048,
        logprobs=True,
        top_logprobs=20
    ),
}


def extract_letter(raw: str, num_choices: int, use_cot: bool = False) -> str:
    """Extract letter answer from model output."""
    raw = raw.strip()
    
    # Valid letters based on number of choices
    valid_letters = [chr(ord('A') + i) for i in range(num_choices)]
    
    if use_cot:
        # In CoT mode, look for answer after "<answer>"
        answer_match = re.search(r'<answer>\s*', raw, re.IGNORECASE)
        if answer_match:
            after_answer = raw[answer_match.end():]
            for char in after_answer:
                if char.upper() in valid_letters:
                    return char.upper()
        return "Cannot parse answer"
    else:
        # Normal mode: Look for first valid letter
        for char in raw:
            if char.upper() in valid_letters:
                return char.upper()
    
    return "Cannot parse answer"


async def get_openai_completion_with_logprobs(
    client: AsyncOpenAI,
    model: str,
    messages: List[Dict[str, str]],
    generation_config: OpenAIGenerationConfig,
    choices: List[str],
    debug: bool = False
) -> Tuple[List[float], str, str, str, Optional[List[Dict]]]:
    """
    Get completion from OpenAI with logprobs for answer choices.
    
    Returns:
        - List of log probabilities for each choice letter (A, B, C, etc.)
        - Raw token text of the predicted answer
        - Parsed letter from the raw token
        - Full generated text
        - Debug info with logprobs (if debug=True)
    """
    # Valid letters based on number of choices
    valid_letters = [chr(ord('A') + i) for i in range(len(choices))]

    try:
        # Prepare API call parameters
        api_params = {
            "model": model,
            "messages": messages,
            "n": 1,
            "temperature": generation_config.temperature,
            "top_p": generation_config.top_p,
            "max_tokens": generation_config.max_tokens,
            "logprobs": generation_config.logprobs,
            "top_logprobs": generation_config.top_logprobs,
        }
        
        # Call OpenAI API
        if debug:
            print(f"Debug: Calling OpenAI API with model {model}")
            print(f"Debug: Messages: {messages}")
            print(f"Debug: API params: {api_params}")
        
        response = await client.chat.completions.create(**api_params)
        
        completion = response.choices[0]
        generated_text = completion.message.content or ""
        
        if debug:
            print(f"Debug: Response content: '{generated_text}'")
            print(f"Debug: Message role: {completion.message.role}")
            print(f"Debug: Finish reason: {completion.finish_reason}")
            if hasattr(response, 'usage'):
                print(f"Debug: Token usage: {response.usage}")
        
        # Check if response is empty
        if not generated_text:
            print(f"Warning: Empty response from {model}. This might be due to content filtering or API issues.")
        
        # Initialize logprobs for each choice
        choice_logprobs = [-100.0] * len(choices)  # Default to very low probability

        if completion.logprobs and completion.logprobs.content:
            # For non-reasoning models with logprobs
            # Find where the answer appears
            found_answer = False
            raw_predicted_token = ""
            parsed_predicted_letter = ""
            debug_tokens = []
            
            # For normal mode, answer should appear right after "Answer:"
            # For CoT mode, answer should appear after "The answer is"
            for i, token_info in enumerate(completion.logprobs.content):
                token = token_info.token
                
                # Check if this token contains a valid letter
                for letter in valid_letters:
                    if letter in token.upper():
                        found_answer = True
                        raw_predicted_token = token
                        parsed_predicted_letter = letter
                        
                        # Get logprobs for all choices from top_logprobs
                        if token_info.top_logprobs:
                            for top_token in token_info.top_logprobs:
                                for j, choice_letter in enumerate(valid_letters):
                                    if choice_letter in top_token.token.upper():
                                        choice_logprobs[j] = top_token.logprob
                                        break
                            
                            # The actual generated token's logprob
                            for j, choice_letter in enumerate(valid_letters):
                                if choice_letter in token.upper():
                                    choice_logprobs[j] = token_info.logprob
                                    break
                        
                        if debug:
                            # Collect debug info
                            debug_tokens = [
                                {
                                    "token": t.token,
                                    "logprob": t.logprob,
                                    "prob": float(2 ** t.logprob)  # Convert to probability
                                }
                                for t in token_info.top_logprobs
                            ] if token_info.top_logprobs else []
                        
                        break
                
                if found_answer:
                    break
            
            if not found_answer:
                # Try to extract from generated text
                parsed_predicted_letter = extract_letter(
                    generated_text, 
                    len(choices), 
                    use_cot=generation_config.suffix == "<answer>"
                )
                raw_predicted_token = parsed_predicted_letter
        
        else:
            # No logprobs available (shouldn't happen for regular models), extract from text
            parsed_predicted_letter = extract_letter(
                generated_text, 
                len(choices), 
                use_cot=generation_config.suffix == "<answer>"
            )
            raw_predicted_token = parsed_predicted_letter
            debug_tokens = None
        
        return choice_logprobs, raw_predicted_token, parsed_predicted_letter, generated_text, debug_tokens
        
    except Exception as e:
        print(f"Error in OpenAI API call: {e}")
        # Return default values on error
        return (
            [-100.0] * len(choices),
            "Error",
            "Cannot parse answer",
            f"Error: {str(e)}",
            None
        )


async def compute_batch_openai_logprobs(
    client: AsyncOpenAI,
    model: str,
    prompts_and_metadata: List[Tuple[List[Dict[str, str]], List[str], Dict]],
    generation_config: OpenAIGenerationConfig,
    max_concurrent: int = 10,
    debug: bool = False
) -> List[Tuple[List[float], str, str, str, Optional[List[Dict]]]]:
    """
    Compute logprobs for a batch of prompts using OpenAI API.
    
    Args:
        client: AsyncOpenAI client
        model: Model name
        prompts_and_metadata: List of (messages, choices, metadata) tuples
        generation_config: Generation configuration
        max_concurrent: Number of requests to process concurrently
        debug: Whether to return debug information
    
    Returns list of tuples containing:
        - logprobs for each choice
        - raw predicted token
        - parsed predicted letter
        - generated text
        - debug info (if debug=True)
    """
    results = []
    
    # Process in batches with rate limiting
    for i in tqdm(range(0, len(prompts_and_metadata), max_concurrent), desc="Processing batches"):
        batch = prompts_and_metadata[i:i + max_concurrent]
        batch_results = []
        
        # Process batch concurrently
        tasks = []
        for messages, choices, metadata in batch:
            task = get_openai_completion_with_logprobs(
                client, model, messages, generation_config, choices, debug
            )
            tasks.append(task)
        
        # Wait for all tasks in batch to complete
        batch_results = await asyncio.gather(*tasks)
        results.extend(batch_results)
    
    return results


def load_model_config(model_key: str) -> str:
    """Load OpenAI model name from config."""
    config_path = ROOT / "configs" / "models_config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    if model_key not in config["models"]:
        raise ValueError(f"Model {model_key} not found in models_config.yaml")
    
    return config["models"][model_key]




async def main():
    parser = argparse.ArgumentParser(description="Compute answers and probabilities using OpenAI API")
    parser.add_argument("--experiment_name", type=str, required=True,
                       help="Name of the experiment (e.g., csqa__gpt_4o__d1nu1nin__nocot)")
    parser.add_argument("--probe_variant", type=str, required=True,
                       help="Probe variant to use (bare, upos, dpos, etc.)")
    parser.add_argument("--eval_jsonl", type=str, required=True,
                       help="Path to evaluation JSONL file")
    parser.add_argument("--output_jsonl", type=str,
                       help="Path to output JSONL file (default: auto-generated)")
    parser.add_argument("--max-concurrent", type=int, default=10,
                       help="Maximum concurrent API calls (default: 10) - NOTE: This value is overridden to 100")
    parser.add_argument("--temperature", type=float,
                       help="Override default temperature")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug output with detailed logprobs")
    parser.add_argument("--api_key", type=str,
                       help="OpenAI API key (default: from OPENAI_API_KEY env var)")
    
    args = parser.parse_args()
    
    # Override max-concurrent with fixed value
    args.max_concurrent = FIXED_MAX_CONCURRENT
    print(f"Using fixed max-concurrent value: {FIXED_MAX_CONCURRENT}")
    
    # Parse experiment name
    exp = parse_experiment_name(args.experiment_name)
    print(f"Parsed experiment: {exp}")
    
    # Check if model is an OpenAI model
    if not exp.model_key.startswith("gpt_"):
        raise ValueError(f"Model {exp.model_key} is not an OpenAI model. Use compute_answers_and_probs.py instead.")
    
    # Load model name from config
    model_name = load_model_config(exp.model_key)
    print(f"Using OpenAI model: {model_name}")

    # Set up OpenAI client
    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key required. Set OPENAI_API_KEY env var or use --api_key")
    
    client = AsyncOpenAI(api_key=api_key)
    
    # Determine generation config based on experiment type
    if exp.use_cot:
        generation_config = OPENAI_GENERATION_CONFIGS["cot"]
        print("Using CoT generation config")
    else:
        generation_config = OPENAI_GENERATION_CONFIGS["normal"]
        print("Using normal generation config")
    
    # Override temperature if specified
    if args.temperature is not None:
        generation_config.temperature = args.temperature
        print(f"Overriding temperature to: {args.temperature}")
    
    # Load evaluation data
    print(f"Loading evaluation data from: {args.eval_jsonl}")
    eval_data = []
    with open(args.eval_jsonl, 'r') as f:
        for line in f:
            eval_data.append(json.loads(line.strip()))
    print(f"Loaded {len(eval_data)} evaluation examples")
    
    # Determine output directory
    if args.output_jsonl:
        output_file = args.output_jsonl
        output_dir = os.path.dirname(output_file)
    else:
        # Auto-generate output path
        output_dir = os.path.join(ROOT, "results", args.experiment_name, "api_outputs")
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"probs_{args.probe_variant}.jsonl")
    
    print(f"Output will be saved to: {output_file}")
    
    # Load canonical wrong answers if needed
    canonical_wrong = {}
    if args.probe_variant != "bare":
        canonical_wrong_file = os.path.join(ROOT, "results", args.experiment_name, "canonical_wrong.jsonl")
        if os.path.exists(canonical_wrong_file):
            print(f"Loading canonical wrong answers from: {canonical_wrong_file}")
            with open(canonical_wrong_file, 'r') as f:
                for line in f:
                    item = json.loads(line.strip())
                    canonical_wrong[item["qid"]] = item["canonical_wrong"]
        else:
            print(f"Warning: Canonical wrong file not found at {canonical_wrong_file}")
    
    # Load tier sentences if needed
    tier_sentences_data = None
    if args.probe_variant != "bare":
        tier_sentences_data = load_tier_sentences(exp.dataset, exp.model_key, exp.instruction, 
                                                exp.use_cot, exp.reasoning_mode)
        if tier_sentences_data:
            print(f"Loaded tier sentences for {len(tier_sentences_data)} questions")
    
    # Prepare prompts
    prompts_and_metadata = []
    
    for item in eval_data:
        qid = item["id"]
        question = item["question"]
        choices = item["choices"]
        gold_letter = item["answerKey"]
        
        # Get wrong answer
        if args.probe_variant == "bare":
            wrong_letter = "A" if gold_letter != "A" else "B"
        else:
            if qid not in canonical_wrong:
                raise ValueError(f"Missing canonical wrong answer for question {qid}")
            wrong_letter = canonical_wrong[qid]
        
        # Get tier sentences if needed
        question_tier_sentences = None
        if args.probe_variant != "bare" and tier_sentences_data:
            tier_data = tier_sentences_data.get(qid)
            if not tier_data:
                raise ValueError(f"Missing tier sentences for question {qid}")
            question_tier_sentences = tier_data
        
        # Extract answer choices
        if isinstance(choices, dict) and "text" in choices:
            answer_choices = choices["text"]
        else:
            answer_choices = choices
        
        # Build messages for OpenAI API
        messages, pure_prompt, system_prompt = build_openai_messages(
            question=question,
            answer_choices=answer_choices,
            exp=exp,
            probe_variant=args.probe_variant,
            gold_answer=gold_letter,
            wrong_answer=wrong_letter,
            tier_sentences=question_tier_sentences,
        )
        
        metadata = {
            "qid": qid,
            "gold": gold_letter,
            "choices": choices,
            "pure_prompt": pure_prompt,
            "messages": messages,
            "probe_variant": args.probe_variant
        }
        
        prompts_and_metadata.append((messages, answer_choices, metadata))
    
    # Compute probabilities
    print(f"Computing probabilities for {len(prompts_and_metadata)} prompts...")
    results = await compute_batch_openai_logprobs(
        client=client,
        model=model_name,
        prompts_and_metadata=prompts_and_metadata,
        generation_config=generation_config,
        max_concurrent=args.max_concurrent,
        debug=args.debug
    )
    
    # Save results
    print(f"Saving results to: {output_file}")
    saved_count = 0

    with open(output_file, 'w') as f:
        for (messages, choices, metadata), (logprobs, raw_token, parsed_letter, generated_text, debug_tokens) in zip(prompts_and_metadata, results):
            # Convert logprobs to probabilities using softmax
            import numpy as np
            logprobs_array = np.array(logprobs)
            # Replace -100 with a very negative number for softmax
            logprobs_array[logprobs_array == -100.0] = -1000.0
            probs = np.exp(logprobs_array - np.max(logprobs_array))  # Numerical stability
            probs = probs / probs.sum()
            probs = probs.tolist()

            # Create letter-based probability results
            letter_probs = []
            for j, (choice, prob, logp) in enumerate(zip(choices, probs, logprobs)):
                letter = chr(ord('A') + j)
                letter_probs.append({
                    "letter": letter,
                    "text": choice,
                    "prob": prob,
                    "logp": float(logp)  # Ensure it's a float
                })
            
            # Extract output letter (what the model actually generated)
            output_letter = parsed_letter
            
            # Determine if model is correct
            output_correct = (output_letter == metadata["gold"])
            
            # Get system prompt from messages
            system_prompt = ""
            for msg in messages:
                if msg["role"] == "system":
                    system_prompt = msg["content"]
                    break
            
            result = {
                "qid": metadata["qid"],
                "probe_variant": args.probe_variant,
                "gold": metadata["gold"],
                "output_letter": output_letter,
                "raw_output": generated_text,
                "output_correct": output_correct,
                "probs": letter_probs,
                "user_prompt": metadata["pure_prompt"],
                "system_prompt": system_prompt,
                "raw_predicted_answer": raw_token,
                "predicted_answer": parsed_letter,
                "generation_suffix": generation_config.suffix,
                "choices": choices,  # Simple list of text strings
                "model": model_name
            }
            
            if args.debug and debug_tokens:
                result["debug_next_tokens"] = debug_tokens
            
            f.write(json.dumps(result) + '\n')
            saved_count += 1
    
    print(f"\nSaved {saved_count} results to {output_file}")
    
    # Compute accuracy
    correct_count = 0
    for i, (messages, choices, metadata) in enumerate(prompts_and_metadata):
        _, _, parsed_letter, _, _ = results[i]
        if parsed_letter == metadata["gold"]:
            correct_count += 1
    
    accuracy = correct_count / len(results) * 100 if results else 0
    print(f"\nAccuracy: {correct_count}/{len(results)} = {accuracy:.2f}%")
    print(f"✓ Completed processing probe variant: {args.probe_variant}")


if __name__ == "__main__":
    asyncio.run(main())