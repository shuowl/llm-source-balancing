#!/usr/bin/env python
"""
core/prompt_style.py
====================

Canonical prompt builder for all phases of the pipeline.

*Experiment name* fixes the template (prior order + strengths).
*Probe variant* (a run-time flag) tells us **which prior(s) to keep**
and whether each of them states the **gold** or the **wrong** answer.

IMPORTANT: Chat Template Handling
---------------------------------
- Qwen3 models (both base and instruct): Have built-in chat templates
- Llama3 base models (e.g., llama3_8b, llama3_1_8b): NO chat template
- Llama3 instruct models (e.g., llama3_8b_instruct): Have chat templates

For Llama base models, we use simple concatenation: system_prompt + "\n\n" + user_prompt
All other models use tokenizer.apply_chat_template()

probe_variant legend
--------------------
single-prior
    bare   → no priors at all
    upos   → user only, asserts gold
    uneg   → user only, asserts wrong
    dpos   → doc  only, asserts gold
    dneg   → doc  only, asserts wrong

double-prior  
    <first><pol><second><pol>
       first ≠ second, first ∈ {u,d}, pol ∈ {p,n}
    Examples (u = user, d = doc, p = positive, n = negative):
       updp  user gold  , doc gold   (user first)
       updn  user gold  , doc wrong  (user first)
       undp  user wrong , doc gold   (user first)
       undn  user wrong , doc wrong  (user first)
       dpup  doc  gold  , user gold  (doc first)
       dpun  doc  gold  , user wrong (doc first)
       dnup  doc  wrong , user gold  (doc first)
       dnun  doc  wrong , user wrong (doc first)
"""

from __future__ import annotations
from typing import Dict, Tuple, Literal, Optional
import json
from pathlib import Path
import logging

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_TIER_DIR = str(_PROJECT_ROOT / "data" / "tier_sentences")

logger = logging.getLogger(__name__)

# ─────────────────── System prompts ──────────────────── #
# System prompts are now generated dynamically in build_prompt() based on the number of choices

_INSTR_APPENDIX = {
    "neutral": "",
    "based_on_docs": (
        " Base your answer ONLY on the information provided in the document statement. "
        "Ignore all other sources including your own knowledge and the user statement."
    ),
    "based_on_user": (
        " Base your answer ONLY on the user's statement. "
        "Ignore all other sources including your own knowledge and any document statement."
    ),
    "own_knowledge_only": (
        " Base your answer ONLY on your own internal knowledge. "
        "Completely ignore any statements from users or documents."
    ),
}

def _make_system_prompt(base: str, variant: str) -> str:
    return (base + _INSTR_APPENDIX[variant]).strip()

# ────────────────── Prior-sentence templates (legacy - kept for backward compatibility) ───────────────── #
USER_TMPL: Dict[str, str] = {
    "weak":      "I'm not entirely sure, but I think the answer might be {ans}.",
    "neutral":   "From my point of view, the answer is {ans}.",
    "confident": "I'm absolutely certain the answer is {ans}.",
}
DOC_TMPL: Dict[str, str] = {
    "weak":      "A single source tentatively mentions that the answer could be {ans}.",
    "neutral":   "Evidence indicates the answer is {ans}.",
    "confident": "Multiple reliable sources state the answer is {ans}.",
}

# ────────────────── Tier sentence loading ───────────────── #
# Cache for loaded tier sentences to avoid repeated file I/O
_TIER_SENTENCES_CACHE = {}

def load_tier_sentences(dataset: str, model_key: str, instruction: str, use_cot: bool, reasoning_mode: bool, tier_dir: str = _DEFAULT_TIER_DIR) -> Optional[Dict[str, Dict]]:
    """Load tier sentences for a dataset-model-variant with caching.
    
    Returns dict mapping question_id to sentences, or None if file not found.
    """
    # Build filename similar to exp_name structure
    # Format: {dataset}__{model_key[r]}__i{instruction}__{cot/nocot}_tier_sentences.json
    model_part = model_key
    if reasoning_mode:
        model_part = model_key + "r"
    cot_mode = "cot" if use_cot else "nocot"
    
    # Check cache first
    cache_key = f"{dataset}__{model_part}__i{instruction}__{cot_mode}"
    if cache_key in _TIER_SENTENCES_CACHE:
        return _TIER_SENTENCES_CACHE[cache_key]
    
    tier_file = Path(tier_dir) / f"{dataset}__{model_part}__i{instruction}__{cot_mode}_tier_sentences.json"
    
    if not tier_file.exists():
        # Only log warning once, not for every call
        if cache_key not in _TIER_SENTENCES_CACHE:
            logger.debug(f"Tier sentences file not found: {tier_file}")
        _TIER_SENTENCES_CACHE[cache_key] = None
        return None
    
    try:
        with open(tier_file, 'r') as f:
            data = json.load(f)
        
        # Convert to dict by question_id for easy lookup
        sentences_by_qid = {}
        for item in data:
            # Handle both formats: with and without "sentences" wrapper
            if "sentences" in item:
                # New format with wrapper
                qid = item["question_id"]
                sentences = item["sentences"]
            else:
                # Old format without wrapper - item itself contains the sentences
                qid = item["question_id"]
                sentences = item
            
            # Clean up any surrounding quotes from tier sentences
            # Example: "\"The report states the answer is singing.\"" -> "The report states the answer is singing."
            cleaned_sentences = {}
            for key, value in sentences.items():
                if key.startswith(('t1_', 't2_')) and isinstance(value, str):
                    # Strip surrounding quotes if present
                    if value.startswith('"') and value.endswith('"') and len(value) > 1:
                        value = value[1:-1]
                    cleaned_sentences[key] = value
                else:
                    cleaned_sentences[key] = value
            
            sentences_by_qid[qid] = cleaned_sentences
        
        # Cache the loaded data
        _TIER_SENTENCES_CACHE[cache_key] = sentences_by_qid
        return sentences_by_qid
    except Exception as e:
        logger.error(f"Error loading tier sentences: {e}")
        _TIER_SENTENCES_CACHE[cache_key] = None
        return None

# ───────────────── Core builder (low-level) ───────────────── #
def build_prompt(
    *,
    tokenizer,
    question: str,
    choices: list[str],
    correct_answer: str,
    wrong_answer: str,
    include_user: bool,
    include_doc: bool,
    user_first: bool,
    user_pos: bool,
    doc_pos: bool,
    user_strength: str,
    doc_strength: str,
    instruction_variant: str,
    enable_thinking: bool,
    use_cot_evaluation: bool,
    doc_label: str = "Document",  # Kept for backward compatibility but not used
    tier_sentences: Optional[Dict] = None,  # Tier sentences for this question
    user_tier: int = 1,  # Which tier to use (1 or 2)
    doc_tier: int = 1,   # Which tier to use (1 or 2)
    exp=None,  # ExpName object to determine model family
    force_answer_prompt: bool = False,  # Force adding "Answer: " even in reasoning mode
    noncommittal_reasoning: bool = False,  # Generate reasoning without committing to answer
    use_normal_system_prompt: bool = False,  # Use normal mode system prompt even in reasoning mode (i.e., use_vllm_reasoning)
) -> Tuple[str, str, str]:
    """Return (chat_prompt, pure_prompt, system_prompt)."""

    # Generate letter options based on number of choices
    letter_options = ", ".join([chr(65+i) for i in range(len(choices))])
    if len(choices) > 1:
        # Replace last comma with "or" for better grammar
        last_comma = letter_options.rfind(", ")
        if last_comma != -1:
            letter_options = letter_options[:last_comma] + " or" + letter_options[last_comma+1:]
    
    # Build system prompt with dynamic letter options
    if use_cot_evaluation:
        base = (
            f"First, think through the problem step-by-step inside a `<think>...</think>` block.\n"
            f"Then, provide ONLY the letter ({letter_options}) inside an `<answer>...</answer>` block.\n\n"
            f"Format:\n"
            f"<think> reasoning process here </think>\n"
            f"<answer> [letter] </answer>\n\n"
            f"Do not write anything outside these tags."
        )
    elif enable_thinking and not use_normal_system_prompt:
        # For models trained to use <think> tags (reasoning mode)
        # Only used for vLLM generation with noncommittal reasoning
        if noncommittal_reasoning:
            # Special instruction for noncommittal reasoning (vLLM generation)
            base = (
                f"Analyze each option ({letter_options}) carefully. "
                f"However, do NOT state your final answer or conclusion in your thinking. "
                f"Just explore the problem without committing to any specific choice."
            )
        else:
            # This case shouldn't happen in the new design
            # All reasoning generation is noncommittal
            raise ValueError("Reasoning mode without noncommittal_reasoning is not supported")
    else:
        # For normal mode (non-reasoning) OR when using normal system prompt in reasoning mode
        base = (
            f"Answer with ONLY the letter ({letter_options}) of your chosen answer. "
            f"Do not include any explanation, punctuation, or additional text."
        )
    
    system_txt = _make_system_prompt(base, instruction_variant)

    # Build prior lines (sources)
    prior_lines: list[str] = []
    
    # Build both sources first
    user_line = None
    doc_line = None
    
    if include_user:
        if not tier_sentences:
            raise ValueError("Tier sentences are required but not provided")
        
        # Use tier sentences
        tier_key = f"t{user_tier}_user_"
        if user_pos:
            user_line = tier_sentences.get(tier_key + "correct")
        else:
            user_line = tier_sentences.get(tier_key + "wrong")
        
        if not user_line:
            raise ValueError(f"Missing tier sentence: {tier_key}{'correct' if user_pos else 'wrong'}")
    
    if include_doc:
        if not tier_sentences:
            raise ValueError("Tier sentences are required but not provided")
        
        # Use tier sentences
        tier_key = f"t{doc_tier}_doc_"
        if doc_pos:
            doc_line = tier_sentences.get(tier_key + "correct")
        else:
            doc_line = tier_sentences.get(tier_key + "wrong")
        
        if not doc_line:
            raise ValueError(f"Missing tier sentence: {tier_key}{'correct' if doc_pos else 'wrong'}")
    
    # Add sources in the specified order
    if include_user and include_doc:
        if user_first:
            prior_lines = [user_line, doc_line]
        else:
            prior_lines = [doc_line, user_line]
    elif include_user:
        prior_lines = [user_line]
    elif include_doc:
        prior_lines = [doc_line]

    # Format answer choices with letters
    formatted_choices = [f"{chr(65+i)}. {choice}" for i, choice in enumerate(choices)]
    
    # Build the prompt with sources first, then question, then choices
    parts = []
    
    # Add sources first if any
    if prior_lines:
        parts.extend(prior_lines)
        parts.append("")  # Empty line after sources
    
    # Add question
    parts.append("Question: " + question)
    parts.append("")  # Empty line before choices
    
    # Add answer choices
    parts.extend(formatted_choices)
    
    # Add empty line after choices for better formatting
    parts.append("")
    
    # Add Answer: prompt at the end based on usage context
    # Logic:
    # - vLLM noncommittal generation (noncommittal_reasoning=True): Don't add "Answer: "
    # - HF probing with vLLM reasoning (use_normal_system_prompt=True): Don't add "Answer: " here (will be added after reasoning)
    # - Everything else: Add "Answer: "
    
    if noncommittal_reasoning:
        # vLLM noncommittal generation - don't add Answer:
        pass
    elif use_normal_system_prompt:
        # HF probing with vLLM reasoning - don't add Answer: here
        # It will be added after the vLLM reasoning in compute_probs_single_variant.py
        pass
    else:
        # Normal HF probing or other use cases - add Answer:
        parts.append("Answer: ")
    
    pure = "\n".join(parts)

    # Build template kwargs
    template_kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    
    # Handle enable_thinking parameter
    # Only set enable_thinking for Qwen3 family models - other models don't support this parameter
    # Check if this is a Qwen3 model by checking if model_key starts with 'qwen3'
    is_qwen3 = exp and exp.model_key and exp.model_key.startswith('qwen3')
    
    if is_qwen3:
        # For Qwen3 models, we must explicitly set enable_thinking
        # Setting it to False prevents thinking (though it adds empty <think></think> tags to template)
        template_kwargs["enable_thinking"] = enable_thinking
    # For non-Qwen3 models, never pass enable_thinking parameter
    
    # Check if this is a base model (not instruction-tuned)
    # Base models need special handling as they don't follow chat templates well
    is_llama_base = (exp and exp.model_key and 
                     exp.model_key.startswith('llama3') and 
                     not exp.model_key.endswith('_instruct'))
    
    is_qwen3_base = (exp and exp.model_key and 
                     exp.model_key.startswith('qwen3') and 
                     exp.model_key.endswith('_base'))
    
    if is_llama_base or is_qwen3_base:
        # Base models don't follow chat templates well
        # Add instruction directly in the prompt for better results
        # This matches the approach in test_base_model_mc_prob.py
        if pure.endswith("Answer: "):
            # Insert instruction before "Answer: "
            pure = pure[:-8]  # Remove "Answer: "
            pure += f"\nPlease respond with only the letter ({letter_options}) of the correct answer.\nAnswer: "
        
        # Use simple concatenation format without chat template
        chat = f"{system_txt}\n\n{pure}"
    else:
        # All other models (including Qwen3 base) have chat templates
        chat = tokenizer.apply_chat_template(
            [{"role": "system", "content": system_txt},
             {"role": "user",   "content": pure}],
            **template_kwargs
        )
    return chat, pure, system_txt

# ─────────────────── Probe-variant wrapper ────────────────── #
ProbeVariant = Literal[
    "bare",
    "upos", "uneg", "dpos", "dneg",
    "updp", "updn", "undp", "undn",
    "dpup", "dpun", "dnup", "dnun",
]

def build_chat_prompt(
    *,
    tokenizer,
    question: str,
    answer_choices: list[str],
    exp,                       # ExpName object
    probe_variant: ProbeVariant,
    gold_answer: str,
    wrong_answer: str,
    doc_label: str = "Document",
    tier_sentences: Optional[Dict] = None,
    force_answer_prompt: bool = False,  # Force adding "Answer: " even in reasoning mode
    noncommittal_reasoning: bool = False,  # Generate reasoning without committing to answer
    use_vllm_reasoning: bool = False,  # Using pre-generated vLLM reasoning for HF probing
):
    """High-level wrapper → returns (chat_prompt, pure_prompt, system_prompt).
    
    Returns:
        chat_prompt: The fully formatted prompt with chat template applied, ready for the model.
            This includes:
            - System message with instructions
            - User message with the question and any prior statements
            - Chat template formatting (tokenizer-specific)
            - Ends with "Answer: " (except in reasoning mode)
        
        pure_prompt: The raw user message content without chat template formatting.
            This contains:
            - Any prior statements (user/doc assertions if included)
            - The actual question
            - Multiple choice options formatted as "A) ..., B) ..., etc."
            - "Answer: " at the end (except in reasoning mode)
        
        system_prompt: The system instruction message that tells the model how to answer:
            - Base instruction about answering with only a letter
            - For CoT: Instructions to use <think> and <answer> tags
            - For reasoning mode: Instructions to reason first then provide letter
            - Additional guidance based on instruction variant (neutral, based_on_docs, 
              based_on_user, own_knowledge_only)
    """

    # Verify template capabilities
    has_user = exp.user_strength is not None
    has_doc  = exp.doc_strength  is not None

    if probe_variant in {"upos", "uneg"} and not has_user:
        raise ValueError("Probe variant requires user prior but template lacks one.")
    if probe_variant in {"dpos", "dneg"} and not has_doc:
        raise ValueError("Probe variant requires doc prior but template lacks one.")

    # Map variant to flags
    include_user = include_doc = user_pos = doc_pos = False
    user_first_runtime = exp.user_first      # default ordering from template

    if probe_variant == "bare":
        pass

    elif probe_variant in {"upos", "uneg"}:
        include_user = True
        user_pos = probe_variant == "upos"

    elif probe_variant in {"dpos", "dneg"}:
        include_doc = True
        doc_pos = probe_variant == "dpos"

    else:  # double-prior
        include_user = include_doc = True

        # Decode polarity by position
        first_pol, second_pol = probe_variant[1], probe_variant[3]
        user_first_runtime = probe_variant[0] == "u"
        user_pos = (probe_variant[0] == "u" and first_pol == "p") or \
                   (probe_variant[2] == "u" and second_pol == "p")
        doc_pos  = (probe_variant[0] == "d" and first_pol == "p") or \
                   (probe_variant[2] == "d" and second_pol == "p")

        # Safety: order must match template
        if user_first_runtime != exp.user_first:
            raise ValueError(
                f"Probe variant '{probe_variant}' assumes "
                f"{'user' if user_first_runtime else 'doc'} first, "
                f"but template is "
                f"{'user' if exp.user_first else 'doc'} first."
            )

    # When using vLLM reasoning for HF probing, we want:
    # - enable_thinking=True (to get thinking tags in template)
    # - use_normal_system_prompt=True (to get normal mode instructions)
    
    chat, pure, system = build_prompt(
        tokenizer=tokenizer,
        question=question,
        choices=answer_choices,
        correct_answer=gold_answer,
        wrong_answer=wrong_answer,
        include_user=include_user,
        include_doc=include_doc,
        user_first=user_first_runtime,
        user_pos=user_pos,
        doc_pos=doc_pos,
        user_strength=exp.user_strength or "neutral",
        doc_strength=exp.doc_strength  or "neutral",
        instruction_variant={
            "n": "neutral",
            "d": "based_on_docs",
            "u": "based_on_user",
            "o": "own_knowledge_only",
        }.get(exp.instruction, "neutral"),
        enable_thinking=exp.reasoning_mode,  # Always use exp.reasoning_mode for template
        use_cot_evaluation=exp.use_cot,
        doc_label=doc_label,
        tier_sentences=tier_sentences,
        user_tier=exp.user_tier,
        doc_tier=exp.doc_tier,
        exp=exp,
        force_answer_prompt=force_answer_prompt,
        noncommittal_reasoning=noncommittal_reasoning,
        use_normal_system_prompt=use_vllm_reasoning,  # Use normal system prompt for HF probing
    )
    return chat, pure, system

def get_prior_templates(experiment_name: str) -> Tuple[str, str]:
    """
    Extract user and doc templates based on experiment name.
    
    Returns:
        Tuple[str, str]: (user_template, doc_template) with appropriate strength
    """
    # Import here to avoid circular imports
    from exp_name import parse_experiment_name
    
    # Parse the experiment name properly
    exp = parse_experiment_name(experiment_name)
    
    user_template = USER_TMPL[exp.user_strength]
    doc_template = DOC_TMPL[exp.doc_strength]
    
    return user_template, doc_template


def load_reasoning_file(
    reasoning_file_path: str,
    eval_jsonl_path: str = None,
    validate_completeness: bool = True,
    backend: str = "vllm"
) -> Dict[str, str]:
    """
    Load and validate reasoning file for a specific probe variant.

    This is a unified function used by all compute_probs scripts to load
    vLLM reasoning with consistent validation.

    Args:
        reasoning_file_path: Path to the reasoning file ({variant}_{backend}_reasoning.jsonl)
        eval_jsonl_path: Path to evaluation data for validation (optional but recommended)
        validate_completeness: Whether to validate that all QIDs from eval are present
        backend: Backend name (e.g. "vllm") for logging purposes
        
    Returns:
        Dict mapping qid to reasoning text (thinking_with_tags field)
        
    Raises:
        FileNotFoundError: If reasoning file doesn't exist
        ValueError: If validation fails (missing QIDs or empty reasoning)
    """
    reasoning_path = Path(reasoning_file_path)
    
    # Check file exists
    if not reasoning_path.exists():
        raise FileNotFoundError(f"{backend.upper()} reasoning file not found: {reasoning_path}")
    
    # Load reasoning entries
    reasoning_map = {}
    with open(reasoning_path) as f:
        lines = f.readlines()
        for line in lines:
            data = json.loads(line)
            qid = data["qid"]
            # Get thinking_with_tags, default to empty string if not present
            val = data.get("thinking_with_tags", "")
            reasoning_map[qid] = val
    
    print(f"Loaded {len(reasoning_map)} {backend.upper()} reasoning entries from {reasoning_path.name}")
    
    # Validate if eval file is provided
    if validate_completeness and eval_jsonl_path:
        eval_path = Path(eval_jsonl_path)
        if eval_path.exists():
            # Load QIDs from eval file
            eval_qids = set()
            with open(eval_path) as f:
                for line in f:
                    data = json.loads(line)
                    eval_qids.add(data["id"])
            
            # Check for missing QIDs
            reasoning_qids = set(reasoning_map.keys())
            missing_qids = eval_qids - reasoning_qids
            
            if missing_qids:
                raise ValueError(
                    f"{backend.upper()} reasoning file incomplete: missing {len(missing_qids)} QIDs out of {len(eval_qids)}. "
                    f"First 5 missing: {list(missing_qids)[:5]}"
                )
            
            # Check for empty reasoning (optional warning)
            empty_count = sum(1 for v in reasoning_map.values() if not v)
            if empty_count > 0:
                print(f"WARNING: {empty_count} QIDs have empty reasoning in {reasoning_path.name}")
    
    return reasoning_map