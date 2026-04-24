#!/usr/bin/env python
"""
core/prompt_style_openai.py
===========================

OpenAI-specific prompt builder for the source-balancing pipeline.

This module is a simplified version of prompt_style.py specifically for OpenAI API usage.
It returns prompts formatted for OpenAI's chat completion API.

Usage:
    from core.prompt_style_openai import build_openai_messages
    
    messages, pure_prompt = build_openai_messages(
        question="What is the capital of France?",
        answer_choices=["London", "Paris", "Berlin", "Madrid"],
        exp=exp,  # ExpName object
        probe_variant="bare",
        gold_answer="B",
        wrong_answer="A",
        tier_sentences=tier_sentences_dict  # Optional
    )
"""

from __future__ import annotations
from typing import Dict, Tuple, List, Literal, Optional
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# ─────────────────── System prompts for OpenAI ──────────────────── #
# Simplified system prompts optimized for OpenAI models

_INSTR_APPENDIX = {
    "neutral": "",
    "based_on_docs": (
        " Base your answer ONLY on the information provided in the document statement. "
        "Ignore all other sources including your own knowledge and the user statement."
    ),
    "based_on_user": (
        " Base your answer ONLY on the user's statement. "
        "Ignore all other sources including your own knowledge and any document information."
    ),
    "own_knowledge_only": (
        " Base your answer ONLY on your own internal knowledge. "
        "Completely ignore any statements from users or documents."
    ),
}

def _make_system_prompt(num_choices: int, instruction_variant: str) -> str:
    """Create system prompt for OpenAI models."""
    # Generate letter options based on number of choices
    letter_options = ", ".join([chr(65+i) for i in range(num_choices)])
    if num_choices > 1:
        # Replace last comma with "or" for better grammar
        last_comma = letter_options.rfind(", ")
        if last_comma != -1:
            letter_options = letter_options[:last_comma] + " or" + letter_options[last_comma+1:]
    
    base = (
        f"Answer with ONLY the letter ({letter_options}) of your chosen answer. "
        f"Do not include any explanation, punctuation, or additional text."
    )
    
    appendix = _INSTR_APPENDIX[instruction_variant]
    return (base + appendix).strip()

# ────────────────── Tier sentence loading ───────────────── #
# Reuse from prompt_style.py
from core.prompt_style import load_tier_sentences

# ───────────────── Core builder for OpenAI ───────────────── #
def build_openai_prompt(
    *,
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
    tier_sentences: Optional[Dict] = None,
    user_tier: int = 1,
    doc_tier: int = 1,
) -> Tuple[List[Dict[str, str]], str, str]:
    """
    Build messages for OpenAI API.
    
    Returns:
        Tuple of (messages, pure_prompt, system_prompt)
        - messages: List of message dicts for OpenAI API
        - pure_prompt: The user content without system prompt
        - system_prompt: The system instruction
    """
    
    # Build system prompt
    system_prompt = _make_system_prompt(len(choices), instruction_variant)
    
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
    
    # Add Answer: prompt at the end
    parts.append("Answer: ")
    
    pure_prompt = "\n".join(parts)
    
    # Create messages array for OpenAI (separate system and user messages)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": pure_prompt}
    ]

    return messages, pure_prompt, system_prompt

# ─────────────────── Probe-variant wrapper ────────────────── #
ProbeVariant = Literal[
    "bare",
    "upos", "uneg", "dpos", "dneg",
    "updp", "updn", "undp", "undn",
    "dpup", "dpun", "dnup", "dnun",
]

def build_openai_messages(
    *,
    question: str,
    answer_choices: list[str],
    exp,                       # ExpName object
    probe_variant: ProbeVariant,
    gold_answer: str,
    wrong_answer: str,
    tier_sentences: Optional[Dict] = None,
) -> Tuple[List[Dict[str, str]], str, str]:
    """
    High-level wrapper for OpenAI prompt building.
    
    Args:
        question: The question text
        answer_choices: List of answer options
        exp: ExpName object containing experiment configuration
        probe_variant: Which probe variant to use
        gold_answer: The correct answer letter
        wrong_answer: The wrong answer letter to use
        tier_sentences: Optional dict of tier sentences for this question
    
    Returns:
        Tuple of (messages, pure_prompt, system_prompt)
        - messages: List of message dicts ready for OpenAI API
        - pure_prompt: The user content without system prompt
        - system_prompt: The system prompt used
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

    messages, pure_prompt, system_prompt = build_openai_prompt(
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
        tier_sentences=tier_sentences,
        user_tier=exp.user_tier,
        doc_tier=exp.doc_tier,
    )
    
    return messages, pure_prompt, system_prompt


# ─────────────────── Utility functions ────────────────── #
def format_openai_prompt_for_display(messages: List[Dict[str, str]]) -> str:
    """
    Format OpenAI messages for human-readable display.
    
    Args:
        messages: List of message dicts from build_openai_messages
    
    Returns:
        Formatted string showing the conversation
    """
    lines = []
    for msg in messages:
        role = msg["role"].upper()
        content = msg["content"]
        lines.append(f"[{role}]")
        lines.append(content)
        lines.append("")  # Empty line between messages
    
    return "\n".join(lines).strip()


def get_answer_from_messages(messages: List[Dict[str, str]]) -> str:
    """
    Extract the expected position where the answer should appear.
    
    This is useful for understanding where in the prompt the model
    should generate the answer.
    
    Args:
        messages: List of message dicts
    
    Returns:
        The suffix that appears before the answer (typically "Answer:")
    """
    if not messages:
        return ""
    
    # Get the last user message
    user_messages = [m for m in messages if m["role"] == "user"]
    if not user_messages:
        return ""
    
    last_user_content = user_messages[-1]["content"]
    
    # The answer is expected after "Answer:"
    if last_user_content.endswith("Answer:"):
        return "Answer:"
    
    return ""