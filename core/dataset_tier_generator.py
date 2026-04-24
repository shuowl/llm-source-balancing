"""
Dataset-aware Tier Generator
============================

Generates T1 and T2 sentences.

Usage:
    # Basic usage (requires canonical_wrong.jsonl from bare probe run)
    python core/dataset_tier_generator.py --config experiments/config.yaml
    
    # With batch processing controls
    python core/dataset_tier_generator.py --config experiments/generated/exp1_config.yaml --batch-size 100 --max-concurrent 100
    
    # Testing with limited questions
    python core/dataset_tier_generator.py --config config.yaml --limit 10
    
    # T1-only mode (no API calls)
    python core/dataset_tier_generator.py --config experiments/generated/exp1_config.yaml --tier1-only

Batch Processing Parameters:
    --batch-size 50: How many questions to load and process as one group
        - Splits your 1000+ questions into chunks of 50
        - Each batch is processed completely before moving to the next
        - Helps manage memory and provides progress checkpoints
    
    --max-concurrent 20: How many API calls can happen simultaneously within each batch
        - Within each batch of 50, only 20 API calls run at the same time
        - The other 30 wait for a slot to open up
        - Controls the rate of API requests to avoid overwhelming the service

Output:
    - Sentences: data/tier_sentences/{dataset}__{model}__i{instruction}__{cot}_tier_sentences.json
    - LLM logs: data/tier_sentences/{dataset}__{model}__i{instruction}__{cot}_llm_responses.jsonl
"""

import json
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import hashlib
from openai import AsyncOpenAI

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_TIER_DIR = str(_PROJECT_ROOT / "data" / "tier_sentences")
_DATASETS_DIR = _PROJECT_ROOT / "data" / "processed_datasets"
try:
    from .llm_client import LLMClient, TierSentenceGenerator
except ImportError:
    # When running as script, use absolute import
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from core.llm_client import LLMClient, TierSentenceGenerator
import logging
from tqdm.asyncio import tqdm_asyncio
from tqdm import tqdm
import random
import time
from collections import defaultdict
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class DatasetTierSentences:
    """Container for all tier sentences for a question"""
    question_id: str
    question: str
    correct_answer: str
    wrong_answer: str
    
    # T1 sentences (bare assertions)
    t1_user_correct: str
    t1_doc_correct: str
    t1_user_wrong: str
    t1_doc_wrong: str
    
    # T2 sentences (contextual phrasing)
    t2_user_correct: str
    t2_doc_correct: str
    t2_user_wrong: str
    t2_doc_wrong: str
    
    # Metadata
    model_used: str = "gpt-4o"
    temperature: float = 0.3
    base_seed: int = 0
    t1_seed: int = 0
    t2_seed: int = 0
    t1_user_template: str = ""
    t1_doc_template: str = ""
    t1_user_words: Optional[dict] = None
    t1_doc_words: Optional[dict] = None
    t2_user_template: str = ""
    t2_doc_template: str = ""
    t2_user_words: Optional[dict] = None
    t2_doc_words: Optional[dict] = None
    
    def to_dict(self) -> dict:
        return asdict(self)


class DatasetTierGenerator:
    """Generate tier sentences for dataset questions"""
    
    
    # T1 Template and Word Pool Definitions
    T1_USER_TEMPLATES = [
        "I {user_verb} the answer is {answer}.",
        "My {user_noun} is that the answer is {answer}.",
        "The answer is {answer}, I {user_verb}."
    ]
    
    T1_DOC_TEMPLATES = [
        "{doc_noun_plural} {doc_verb} the answer is {answer}.",
        "The {doc_noun} {doc_verb_singular} the answer is {answer}.",
        "The answer is {answer}, {doc_noun_plural_lower} {doc_verb}."
    ]
    
    T1_USER_WORDS = {
        "user_verb": ["think", "believe", "suppose", "reckon", "assume"],
        "user_noun": ["belief", "opinion", "view", "impression", "understanding"]
    }
    
    T1_DOC_WORDS = {
        "doc_verb": ["indicate", "suggest", "state", "note", "mention", "say"],
        "doc_verb_singular": ["states", "indicates", "says", "notes", "mentions"],
        "doc_noun": ["record", "document", "source", "report", "material"],
        "doc_noun_plural": ["Records", "Documents", "Sources", "Reports", "Materials"],
        "doc_noun_plural_lower": ["records", "documents", "sources", "reports", "materials"]
    }
    
    # Math-specific T1 templates and words for datasets like gsm8k
    T1_MATH_USER_TEMPLATES = [
        "I {user_verb} the answer is {answer}.",
        "My {user_noun} is that the answer is {answer}.",
        "The answer is {answer}, I {user_verb}."
    ]
    
    T1_MATH_DOC_TEMPLATES = [
        "{doc_noun_plural} {doc_verb} the answer is {answer}.",
        "The {doc_noun} {doc_verb_singular} the answer is {answer}.",
        "The answer is {answer}, {doc_noun_plural_lower} {doc_verb}."
    ]
    
    T1_MATH_USER_WORDS = {
        "user_verb": ["calculate", "determine", "estimate", "compute", "figure"],
        "user_noun": ["calculation", "estimate", "computation", "assessment", "determination"]
    }
    
    T1_MATH_DOC_WORDS = {
        "doc_verb": ["show", "demonstrate", "indicate", "specify", "present", "reveal"],
        "doc_verb_singular": ["shows", "demonstrates", "indicates", "specifies", "presents"],
        "doc_noun": ["calculation", "solution", "analysis", "method", "result"],
        "doc_noun_plural": ["Calculations", "Solutions", "Analyses", "Methods", "Results"],
        "doc_noun_plural_lower": ["calculations", "solutions", "analyses", "methods", "results"]
    }
    
    # T2 Template and Word Pool Definitions
    T2_USER_TEMPLATES = [
        "{user_phrase} {contextual_claim}.",
        "{contextual_claim}, from what {user_phrase}.",
        "Based on {user_perspective}, {contextual_claim}."
    ]
    
    T2_DOC_TEMPLATES = [
        "{doc_phrase} {contextual_claim}.",
        "{contextual_claim}, according to {doc_source}.",
        "As per {doc_source}, {contextual_claim}."
    ]
    
    T2_USER_WORDS = {
        "user_phrase": ["I've noticed", "I've seen", "I've heard", "I recall", "I've observed", "I believe", "I think"],
        "user_perspective": ["my experience", "my understanding", "what I've seen", "my observation"]
    }
    
    T2_DOC_WORDS = {
        "doc_phrase": ["Studies suggest", "Papers indicate", "Documents suggest", "Sources mention", "Reports note"],
        "doc_source": ["recent reports", "available data", "published studies", "the literature", "the documentation", "recent findings", "available materials", "available sources"]
    }
    
    # CodeMMLU-specific T1 templates and words
    T1_CODEMMLU_USER_TEMPLATES = [
        "I {user_verb} the answer is {answer}.",
        "My {user_noun} is that the answer is {answer}.",
        "The answer is {answer}, I {user_verb}."
    ]
    
    T1_CODEMMLU_DOC_TEMPLATES = [
        "{doc_noun_plural} {doc_verb} the answer is {answer}.",
        "The {doc_noun} {doc_verb_singular} the answer is {answer}.",
        "The answer is {answer}, {doc_noun_plural_lower} {doc_verb}."
    ]
    
    T1_CODEMMLU_USER_WORDS = {
        "user_verb": ["deduce", "conclude", "determine", "assess", "infer"],
        "user_noun": ["assessment", "conclusion", "analysis", "understanding", "interpretation"]
    }
    
    T1_CODEMMLU_DOC_WORDS = {
        "doc_verb": ["indicate", "show", "state", "confirm", "demonstrate", "suggest"],
        "doc_verb_singular": ["indicates", "shows", "states", "confirms", "suggests"],
        "doc_noun": ["reference", "documentation", "resource", "material", "source"],
        "doc_noun_plural": ["References", "Documentation", "Resources", "Materials", "Sources"],
        "doc_noun_plural_lower": ["references", "documentation", "resources", "materials", "sources"]
    }
    
    # Math-specific T2 templates and words for datasets like gsm8k
    T2_MATH_USER_TEMPLATES = [
        "{user_phrase} {contextual_claim}.",
        "{contextual_claim}, from what {user_phrase}.",
        "Based on {user_perspective}, {contextual_claim}."
    ]
    
    T2_MATH_DOC_TEMPLATES = [
        "{doc_phrase} {contextual_claim}.",
        "{contextual_claim}, according to {doc_source}.",
        "As per {doc_source}, {contextual_claim}."
    ]
    
    T2_MATH_USER_WORDS = {
        "user_phrase": ["I've calculated", "I've worked out", "I've computed", "I've solved", "I've derived", "I've determined"],
        "user_perspective": ["my calculations", "my workings", "my analysis", "my solution approach"]
    }
    
    T2_MATH_DOC_WORDS = {
        "doc_phrase": ["Calculations show", "Solutions indicate", "Analysis reveals", "Methods demonstrate", "Results confirm"],
        "doc_source": ["the calculations", "the solution method", "the analysis", "the mathematical approach", "the problem solution", "the computational results"]
    }
    
    # CodeMMLU-specific T2 templates and words
    T2_CODEMMLU_USER_TEMPLATES = [
        "{user_phrase} {contextual_claim}.",
        "{contextual_claim}, from what {user_phrase}.",
        "Based on {user_perspective}, {contextual_claim}."
    ]
    
    T2_CODEMMLU_DOC_TEMPLATES = [
        "{doc_phrase} {contextual_claim}.",
        "{contextual_claim}, according to {doc_source}.",
        "As per {doc_source}, {contextual_claim}."
    ]
    
    T2_CODEMMLU_USER_WORDS = {
        "user_phrase": ["I've analyzed", "I've examined", "I've reviewed", "I've studied", "I've evaluated", "I've investigated"],
        "user_perspective": ["my analysis", "my examination", "my review", "my evaluation"]
    }
    
    T2_CODEMMLU_DOC_WORDS = {
        "doc_phrase": ["Documentation states", "References indicate", "Resources show", "Technical materials confirm", "Sources reveal"],
        "doc_source": ["the documentation", "technical references", "available resources", "reference materials", "technical sources", "the reference guide"]
    }
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o",
        temperature: float = 0.3,
        tier1_only: bool = False,
        log_dir: Optional[str] = None
    ):
        import os
        self.tier1_only = tier1_only
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        
        # Only require API key if not in tier1_only mode
        if not self.tier1_only and not self.api_key:
            raise ValueError("OpenAI API key required for T2 generation. Set OPENAI_API_KEY env var.")
        
        # Initialize LLM client if we have an API key
        if self.api_key and not self.tier1_only:
            # Use tier_sentences directory and experiment-specific log filename
            if log_dir:
                self.log_dir = Path(log_dir)
            else:
                self.log_dir = Path(_DEFAULT_TIER_DIR)
            
            self.llm_client = LLMClient(
                api_key=self.api_key,
                model=model,
                temperature=temperature,
                log_dir=str(self.log_dir)
            )
            self.tier_generator = TierSentenceGenerator(self.llm_client)
        else:
            self.llm_client = None
            self.tier_generator = None
            self.log_dir = None
            
        self.model = model
        self.temperature = temperature
    
    def _get_seeds(self, base_seed: Optional[int] = None) -> Tuple[int, int, int]:
        """Generate seeds for T1 and T2 sampling from base seed"""
        if base_seed is None:
            base_seed = int(time.time() * 1000000) % (2**32)  # Use current time in microseconds
        
        # Create derived seeds for T1 and T2
        t1_seed = (base_seed * 2654435761) % (2**32)  # Large prime multiplier
        t2_seed = (base_seed * 3141592653) % (2**32)  # Another large prime
        
        return base_seed, t1_seed, t2_seed
    
    def _generate_independent_seeds(self, base_seed: int, num_seeds: int) -> List[int]:
        """Generate multiple independent seeds from a base seed"""
        # List of large primes for seed generation
        primes = [
            2654435761, 3141592653, 1103515245, 134775813,
            1664525, 214013, 742938285, 1013904223,
            1284865837, 1481765933, 1698661357, 1966079973
        ]
        
        seeds = []
        for i in range(num_seeds):
            # Use different prime for each seed
            prime = primes[i % len(primes)]
            # Add index to ensure uniqueness even if we cycle through primes
            seed = ((base_seed + i) * prime) % (2**32)
            seeds.append(seed)
        
        return seeds
    
    def _sample_t1_elements(self, seed: int, is_math_dataset: bool = False, is_code_dataset: bool = False) -> Tuple[str, str, dict, dict]:
        """Sample T1 templates and words using given seed with derived sub-seeds"""
        # Select appropriate templates and words based on dataset type
        if is_code_dataset:
            user_templates = self.T1_CODEMMLU_USER_TEMPLATES
            doc_templates = self.T1_CODEMMLU_DOC_TEMPLATES
            user_words_pool = self.T1_CODEMMLU_USER_WORDS
            doc_words_pool = self.T1_CODEMMLU_DOC_WORDS
        elif is_math_dataset:
            user_templates = self.T1_MATH_USER_TEMPLATES
            doc_templates = self.T1_MATH_DOC_TEMPLATES
            user_words_pool = self.T1_MATH_USER_WORDS
            doc_words_pool = self.T1_MATH_DOC_WORDS
        else:
            user_templates = self.T1_USER_TEMPLATES
            doc_templates = self.T1_DOC_TEMPLATES
            user_words_pool = self.T1_USER_WORDS
            doc_words_pool = self.T1_DOC_WORDS
        
        # Derive independent seeds for each sampling decision
        seed_offset = 0
        user_template_seed = seed + seed_offset
        seed_offset += 1
        doc_template_seed = seed + seed_offset
        seed_offset += 1
        
        # Sample templates with independent seeds
        user_template_rng = random.Random(user_template_seed)
        user_template = user_template_rng.choice(user_templates)
        
        doc_template_rng = random.Random(doc_template_seed)
        doc_template = doc_template_rng.choice(doc_templates)
        
        # Sample words for user template - each word gets its own seed
        user_words = {}
        if "{user_verb}" in user_template:
            user_verb_rng = random.Random(seed + seed_offset)
            seed_offset += 1
            user_words["user_verb"] = user_verb_rng.choice(user_words_pool["user_verb"])
        if "{user_noun}" in user_template:
            user_noun_rng = random.Random(seed + seed_offset)
            seed_offset += 1
            user_words["user_noun"] = user_noun_rng.choice(user_words_pool["user_noun"])
        
        # Sample words for doc template - each word gets its own seed
        doc_words = {}
        if "{doc_verb}" in doc_template:
            doc_verb_rng = random.Random(seed + seed_offset)
            seed_offset += 1
            doc_words["doc_verb"] = doc_verb_rng.choice(doc_words_pool["doc_verb"])
        if "{doc_verb_singular}" in doc_template:
            doc_verb_singular_rng = random.Random(seed + seed_offset)
            seed_offset += 1
            doc_words["doc_verb_singular"] = doc_verb_singular_rng.choice(doc_words_pool["doc_verb_singular"])
        if "{doc_noun}" in doc_template:
            doc_noun_rng = random.Random(seed + seed_offset)
            seed_offset += 1
            doc_words["doc_noun"] = doc_noun_rng.choice(doc_words_pool["doc_noun"])
        if "{doc_noun_plural}" in doc_template:
            doc_noun_plural_rng = random.Random(seed + seed_offset)
            seed_offset += 1
            doc_words["doc_noun_plural"] = doc_words_pool["doc_noun_plural"][doc_noun_plural_rng.randrange(len(doc_words_pool["doc_noun_plural"]))]
        if "{doc_noun_plural_lower}" in doc_template:
            doc_noun_plural_lower_rng = random.Random(seed + seed_offset)
            seed_offset += 1
            doc_words["doc_noun_plural_lower"] = doc_words_pool["doc_noun_plural_lower"][doc_noun_plural_lower_rng.randrange(len(doc_words_pool["doc_noun_plural_lower"]))]
        
        return user_template, doc_template, user_words, doc_words
    
    def _generate_t1_sentences(
        self, 
        correct_answer: str, 
        wrong_answer: str,
        user_template: str,
        doc_template: str,
        user_words: dict,
        doc_words: dict,
        is_code_dataset: bool = False
    ) -> Tuple[str, str, str, str]:
        """Generate T1 sentences locally without LLM"""
        # For CodeMMLU, convert answers to lowercase
        if is_code_dataset:
            correct_answer = correct_answer.lower()
            wrong_answer = wrong_answer.lower()
        
        # Generate user sentences
        t1_user_correct = user_template.format(answer=correct_answer, **user_words)
        t1_user_wrong = user_template.format(answer=wrong_answer, **user_words)
        
        # Generate doc sentences
        t1_doc_correct = doc_template.format(answer=correct_answer, **doc_words)
        t1_doc_wrong = doc_template.format(answer=wrong_answer, **doc_words)
        
        return t1_user_correct, t1_doc_correct, t1_user_wrong, t1_doc_wrong
    
    def _sample_t2_elements(self, seed: int, is_math_dataset: bool = False, is_code_dataset: bool = False) -> Tuple[str, str, dict, dict]:
        """Sample T2 templates and words using given seed with derived sub-seeds"""
        # Select appropriate templates and words based on dataset type
        if is_code_dataset:
            user_templates = self.T2_CODEMMLU_USER_TEMPLATES
            doc_templates = self.T2_CODEMMLU_DOC_TEMPLATES
            user_words_pool = self.T2_CODEMMLU_USER_WORDS
            doc_words_pool = self.T2_CODEMMLU_DOC_WORDS
        elif is_math_dataset:
            user_templates = self.T2_MATH_USER_TEMPLATES
            doc_templates = self.T2_MATH_DOC_TEMPLATES
            user_words_pool = self.T2_MATH_USER_WORDS
            doc_words_pool = self.T2_MATH_DOC_WORDS
        else:
            user_templates = self.T2_USER_TEMPLATES
            doc_templates = self.T2_DOC_TEMPLATES
            user_words_pool = self.T2_USER_WORDS
            doc_words_pool = self.T2_DOC_WORDS
        
        # Derive independent seeds for each sampling decision
        seed_offset = 0
        user_template_seed = seed + seed_offset
        seed_offset += 1
        doc_template_seed = seed + seed_offset
        seed_offset += 1
        
        # Sample templates with independent seeds
        user_template_rng = random.Random(user_template_seed)
        user_template = user_template_rng.choice(user_templates)
        
        doc_template_rng = random.Random(doc_template_seed)
        doc_template = doc_template_rng.choice(doc_templates)
        
        # Sample words for user template - each word gets its own seed
        user_words = {}
        if "{user_phrase}" in user_template:
            user_phrase_rng = random.Random(seed + seed_offset)
            seed_offset += 1
            user_words["user_phrase"] = user_phrase_rng.choice(user_words_pool["user_phrase"])
        if "{user_perspective}" in user_template:
            user_perspective_rng = random.Random(seed + seed_offset)
            seed_offset += 1
            user_words["user_perspective"] = user_perspective_rng.choice(user_words_pool["user_perspective"])
        
        # Sample words for doc template - each word gets its own seed
        doc_words = {}
        if "{doc_phrase}" in doc_template:
            doc_phrase_rng = random.Random(seed + seed_offset)
            seed_offset += 1
            doc_words["doc_phrase"] = doc_phrase_rng.choice(doc_words_pool["doc_phrase"])
        if "{doc_source}" in doc_template:
            doc_source_rng = random.Random(seed + seed_offset)
            seed_offset += 1
            doc_words["doc_source"] = doc_source_rng.choice(doc_words_pool["doc_source"])
        
        return user_template, doc_template, user_words, doc_words
    
    async def generate_all_tiers_with_retry(
        self,
        question_id: str,
        question: str,
        correct_answer: str,
        wrong_answer: str,
        dataset_name: str = "",
        max_retries: int = 5
    ) -> Tuple[DatasetTierSentences, int]:
        """
        Generate all tier sentences with retry logic for T2 validation
        Returns tuple of (sentences, attempts_taken)
        """
        for attempt in range(1, max_retries + 1):
            try:
                sentences = await self.generate_all_tiers(
                    question_id, question, correct_answer, wrong_answer, dataset_name
                )
                
                # Validate T2 sentences (skip if tier1_only)
                if self.tier1_only:
                    validation_errors = {}
                else:
                    validation_errors = self._validate_t2_sentences(sentences)
                
                if not validation_errors:
                    # Success!
                    logger.info(f"Question {question_id}: T2 sentences validated successfully on attempt {attempt}")
                    return sentences, attempt
                else:
                    # Log validation errors
                    logger.warning(f"Question {question_id} attempt {attempt}: T2 validation failed")
                    for field, errors in validation_errors.items():
                        for error in errors:
                            logger.warning(f"  {field}: {error}")
                    
                    if attempt < max_retries:
                        logger.info(f"Retrying question {question_id} (attempt {attempt + 1}/{max_retries})...")
                        await asyncio.sleep(1)  # Brief pause before retry
                    else:
                        logger.error(f"Question {question_id}: Failed validation after {max_retries} attempts")
                        return sentences, attempt  # Return the last attempt even if failed
                        
            except Exception as e:
                logger.error(f"Question {question_id} attempt {attempt}: Generation failed with error: {e}")
                if attempt < max_retries:
                    await asyncio.sleep(1)
                else:
                    raise
        
        # Should not reach here, but just in case
        raise Exception(f"Failed to generate sentences for {question_id} after {max_retries} attempts")
    
    async def generate_all_tiers(
        self,
        question_id: str,
        question: str,
        correct_answer: str,
        wrong_answer: str,
        dataset_name: str = ""
    ) -> DatasetTierSentences:
        """Generate all tier sentences - T1 locally, T2 via GPT-4o"""
        
        # Check dataset type
        is_math_dataset = dataset_name.lower() in ['gsm8k', 'mathqa', 'aqua', 'math', 'asdiv']
        is_code_dataset = dataset_name.lower() in ['codemmlu']
        
        # Generate seeds
        base_seed, t1_seed, t2_seed = self._get_seeds()
        
        # Generate T1 sentences locally
        t1_user_template, t1_doc_template, t1_user_words, t1_doc_words = self._sample_t1_elements(t1_seed, is_math_dataset, is_code_dataset)
        t1_user_correct, t1_doc_correct, t1_user_wrong, t1_doc_wrong = self._generate_t1_sentences(
            correct_answer, wrong_answer, t1_user_template, t1_doc_template, t1_user_words, t1_doc_words, is_code_dataset
        )
        
        # If tier1_only, skip T2 generation
        if self.tier1_only:
            # Use empty strings for T2 sentences
            t2_sentences = {
                "t2_user_correct": "",
                "t2_doc_correct": "",
                "t2_user_wrong": "",
                "t2_doc_wrong": ""
            }
            t2_user_template = ""
            t2_doc_template = ""
            t2_user_words = {}
            t2_doc_words = {}
        else:
            # Sample T2 elements
            t2_user_template, t2_doc_template, t2_user_words, t2_doc_words = self._sample_t2_elements(t2_seed, is_math_dataset, is_code_dataset)
            
            try:
                # Use the new tier generator
                t2_sentences = await self.tier_generator.generate_t2_sentences(
                    question=question,
                    correct_answer=correct_answer,
                    wrong_answer=wrong_answer,
                    user_template=t2_user_template,
                    doc_template=t2_doc_template,
                    user_words=t2_user_words,
                    doc_words=t2_doc_words,
                    is_math_dataset=is_math_dataset,
                    is_code_dataset=is_code_dataset,
                    metadata={
                        "question_id": question_id,
                        "dataset_name": dataset_name,
                        "t2_seed": t2_seed
                    }
                )
                
            except Exception as e:
                logger.error(f"Failed to generate T2 sentences for {question_id}: {e}")
                raise
        
        return DatasetTierSentences(
                question_id=question_id,
                question=question,
                correct_answer=correct_answer,
                wrong_answer=wrong_answer,
                t1_user_correct=t1_user_correct,
                t1_doc_correct=t1_doc_correct,
                t1_user_wrong=t1_user_wrong,
                t1_doc_wrong=t1_doc_wrong,
                **t2_sentences,
                model_used=self.model,
                temperature=self.temperature,
                base_seed=base_seed,
                t1_seed=t1_seed,
                t2_seed=t2_seed,
                t1_user_template=t1_user_template,
                t1_doc_template=t1_doc_template,
                t1_user_words=t1_user_words,
                t1_doc_words=t1_doc_words,
                t2_user_template=t2_user_template,
                t2_doc_template=t2_doc_template,
                t2_user_words=t2_user_words,
                t2_doc_words=t2_doc_words
            )
    
    
    def _validate_t2_sentences(self, sentences_data: DatasetTierSentences) -> Dict[str, List[str]]:
        """
        Validate that T2 sentences contain the appropriate answers
        Returns dict with field names that have errors
        """
        errors = {}
        
        # Check each T2 field
        t2_expectations = {
            "t2_user_correct": sentences_data.correct_answer,
            "t2_doc_correct": sentences_data.correct_answer,
            "t2_user_wrong": sentences_data.wrong_answer,
            "t2_doc_wrong": sentences_data.wrong_answer
        }
        
        for field, expected_answer in t2_expectations.items():
            sentence = getattr(sentences_data, field)
            
            # Check if the expected answer is in the sentence
            if not self._contains_answer(sentence, expected_answer):
                if field not in errors:
                    errors[field] = []
                errors[field].append(f"Expected '{expected_answer}' not found in: {sentence}")
                
                # Also check if it contains the wrong answer
                other_answer = sentences_data.wrong_answer if expected_answer == sentences_data.correct_answer else sentences_data.correct_answer
                if self._contains_answer(sentence, other_answer):
                    errors[field].append(f"Contains wrong answer '{other_answer}' instead")
        
        return errors
    
    def _contains_answer(self, sentence: str, answer: str) -> bool:
        """Check if sentence contains the answer (case-insensitive)"""
        import re
        
        # First try exact match
        if answer.lower() in sentence.lower():
            return True
        
        # Try with word boundaries
        pattern = r'\b' + re.escape(answer) + r'\b'
        if re.search(pattern, sentence, re.IGNORECASE):
            return True
        
        # Handle numbers that might be written as words (simplified check)
        if answer.isdigit():
            return answer in sentence
        
        return False
    
    async def generate_batch_tiers(
        self,
        questions_batch: List[Dict[str, Any]],
        max_concurrent: int = 20,
        progress_callback: Optional[callable] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate tier sentences for a batch of questions concurrently
        
        Args:
            questions_batch: List of question data dicts
            max_concurrent: Max concurrent API calls
            progress_callback: Optional progress callback
            
        Returns:
            List of results with sentences and metadata
        """
        if self.tier1_only:
            # For tier1-only, process synchronously since no API calls
            results = []
            for i, q_data in enumerate(questions_batch):
                try:
                    base_seed, t1_seed, t2_seed = self._get_seeds()
                    is_math_dataset = q_data.get("is_math_dataset", False)
                    is_code_dataset = q_data.get("is_code_dataset", False)
                    
                    # Generate T1 sentences locally
                    t1_user_template, t1_doc_template, t1_user_words, t1_doc_words = self._sample_t1_elements(t1_seed, is_math_dataset, is_code_dataset)
                    t1_user_correct, t1_doc_correct, t1_user_wrong, t1_doc_wrong = self._generate_t1_sentences(
                        q_data["correct_answer"], q_data["wrong_answer"], 
                        t1_user_template, t1_doc_template, t1_user_words, t1_doc_words, is_code_dataset
                    )
                    
                    sentences = DatasetTierSentences(
                        question_id=q_data["question_id"],
                        question=q_data["question"],
                        correct_answer=q_data["correct_answer"],
                        wrong_answer=q_data["wrong_answer"],
                        t1_user_correct=t1_user_correct,
                        t1_doc_correct=t1_doc_correct,
                        t1_user_wrong=t1_user_wrong,
                        t1_doc_wrong=t1_doc_wrong,
                        t2_user_correct="",
                        t2_doc_correct="",
                        t2_user_wrong="",
                        t2_doc_wrong="",
                        model_used=self.model,
                        temperature=self.temperature,
                        base_seed=base_seed,
                        t1_seed=t1_seed,
                        t2_seed=t2_seed,
                        t1_user_template=t1_user_template,
                        t1_doc_template=t1_doc_template,
                        t1_user_words=t1_user_words,
                        t1_doc_words=t1_doc_words,
                        t2_user_template="",
                        t2_doc_template="",
                        t2_user_words={},
                        t2_doc_words={}
                    )
                    
                    results.append({
                        "sentences": sentences,
                        "error": None,
                        "metadata": q_data
                    })
                except Exception as e:
                    results.append({
                        "sentences": None,
                        "error": str(e),
                        "metadata": q_data
                    })
                
                if progress_callback:
                    progress_callback(i + 1, len(questions_batch))
            
            return results
        
        # For full tier generation, prepare prompts and use batch processing
        prompts_data = []
        
        for q_data in questions_batch:
            try:
                base_seed, t1_seed, t2_seed = self._get_seeds()
                is_math_dataset = q_data.get("is_math_dataset", False)
                is_code_dataset = q_data.get("is_code_dataset", False)
                
                # Generate T1 locally first
                t1_user_template, t1_doc_template, t1_user_words, t1_doc_words = self._sample_t1_elements(t1_seed, is_math_dataset, is_code_dataset)
                t1_user_correct, t1_doc_correct, t1_user_wrong, t1_doc_wrong = self._generate_t1_sentences(
                    q_data["correct_answer"], q_data["wrong_answer"],
                    t1_user_template, t1_doc_template, t1_user_words, t1_doc_words, is_code_dataset
                )
                
                # Sample T2 elements
                t2_user_template, t2_doc_template, t2_user_words, t2_doc_words = self._sample_t2_elements(t2_seed, is_math_dataset, is_code_dataset)
                
                # Build prompt for T2
                prompt = self.tier_generator._build_t2_prompt(
                    q_data["question"],
                    q_data["correct_answer"],
                    q_data["wrong_answer"],
                    t2_user_template,
                    t2_doc_template,
                    t2_user_words,
                    t2_doc_words,
                    is_math_dataset,
                    is_code_dataset
                )
                
                prompts_data.append({
                    "prompt": prompt,
                    "metadata": {
                        **q_data,
                        "base_seed": base_seed,
                        "t1_seed": t1_seed,
                        "t2_seed": t2_seed,
                        "t1_user_template": t1_user_template,
                        "t1_doc_template": t1_doc_template,
                        "t1_user_words": t1_user_words,
                        "t1_doc_words": t1_doc_words,
                        "t1_user_correct": t1_user_correct,
                        "t1_doc_correct": t1_doc_correct,
                        "t1_user_wrong": t1_user_wrong,
                        "t1_doc_wrong": t1_doc_wrong,
                        "t2_user_template": t2_user_template,
                        "t2_doc_template": t2_doc_template,
                        "t2_user_words": t2_user_words,
                        "t2_doc_words": t2_doc_words
                    }
                })
            except Exception as e:
                logger.error(f"Error preparing prompt for {q_data.get('question_id')}: {e}")
                prompts_data.append({
                    "prompt": "",
                    "metadata": q_data,
                    "error": str(e)
                })
        
        # Process batch with LLM
        batch_results = await self.llm_client.generate_batch(
            prompts_data,
            max_concurrent=max_concurrent,
            progress_callback=progress_callback
        )
        
        # Parse results and create DatasetTierSentences objects
        final_results = []
        for result in batch_results:
            metadata = result.get("metadata", {})
            
            if result.get("error") or metadata.get("error"):
                final_results.append({
                    "sentences": None,
                    "error": result.get("error") or metadata.get("error"),
                    "metadata": metadata
                })
                continue
            
            try:
                # Parse T2 response
                t2_sentences = self.tier_generator._parse_t2_response(result["response"])
                
                # Create full sentences object
                sentences = DatasetTierSentences(
                    question_id=metadata["question_id"],
                    question=metadata["question"],
                    correct_answer=metadata["correct_answer"],
                    wrong_answer=metadata["wrong_answer"],
                    t1_user_correct=metadata["t1_user_correct"],
                    t1_doc_correct=metadata["t1_doc_correct"],
                    t1_user_wrong=metadata["t1_user_wrong"],
                    t1_doc_wrong=metadata["t1_doc_wrong"],
                    **t2_sentences,
                    model_used=self.model,
                    temperature=self.temperature,
                    base_seed=metadata["base_seed"],
                    t1_seed=metadata["t1_seed"],
                    t2_seed=metadata["t2_seed"],
                    t1_user_template=metadata["t1_user_template"],
                    t1_doc_template=metadata["t1_doc_template"],
                    t1_user_words=metadata["t1_user_words"],
                    t1_doc_words=metadata["t1_doc_words"],
                    t2_user_template=metadata["t2_user_template"],
                    t2_doc_template=metadata["t2_doc_template"],
                    t2_user_words=metadata["t2_user_words"],
                    t2_doc_words=metadata["t2_doc_words"]
                )
                
                final_results.append({
                    "sentences": sentences,
                    "error": None,
                    "metadata": metadata
                })
            except Exception as e:
                logger.error(f"Error parsing response for {metadata.get('question_id')}: {e}")
                final_results.append({
                    "sentences": None,
                    "error": str(e),
                    "metadata": metadata
                })
        
        return final_results
    


async def generate_dataset_tiers(
    dataset_name: str,
    model_key: str,
    instruction: str,
    cot_mode: str,
    reasoning_mode: bool,
    canonical_wrong_file: str,
    limit: Optional[int] = None,
    output_dir: str = _DEFAULT_TIER_DIR,
    tier1_only: bool = False,
    batch_size: int = 50,
    max_concurrent: int = 20
) -> None:
    """
    Generate tier sentences for a dataset
    
    Args:
        dataset_name: Name of dataset (e.g., 'csqa')
        model_key: Model key from experiment (e.g., 'qwen3_1_7b')
        instruction: Instruction mode ('n' or 'i')
        cot_mode: Chain-of-thought mode ('cot' or 'nocot')
        reasoning_mode: Whether reasoning mode is enabled
        canonical_wrong_file: Path to canonical_wrong.jsonl (required)
        limit: Limit number of questions to process from start (not from where left off)
        output_dir: Where to save generated sentences
        tier1_only: If True, only generate T1 sentences (no API calls)
        batch_size: Number of questions to process per batch
        max_concurrent: Maximum concurrent API calls per batch
    """
    # Build filename parts similar to exp_name.py structure
    # Format: {dataset}__{model_key[r]}__{instruction}n__{cot/nocot}_tier_sentences.json
    model_part = model_key
    if reasoning_mode:
        model_part = model_key + "r"
    
    # Log what we're processing
    logger.info(f"\n{'='*60}")
    logger.info(f"TIER SENTENCE GENERATION REQUEST")
    logger.info(f"{'='*60}")
    logger.info(f"Dataset: {dataset_name}")
    logger.info(f"Model: {model_key} {'(reasoning)' if reasoning_mode else ''}")
    logger.info(f"Instruction: {instruction}")
    logger.info(f"CoT mode: {cot_mode}")
    logger.info(f"{'='*60}\n")
    
    # Check if output file already exists
    output_path = Path(output_dir)
    output_file = output_path / f"{dataset_name}__{model_part}__i{instruction}__{cot_mode}_tier_sentences.json"
    
    existing_results = []
    processed_question_ids = set()
    
    if output_file.exists():
        logger.info(f"\n{'='*60}")
        logger.info(f"EXISTING OUTPUT FOUND")
        logger.info(f"File: {output_file}")
        
        with open(output_file, 'r') as f:
            existing_results = json.load(f)
        
        # Extract already processed question IDs
        for result in existing_results:
            processed_question_ids.add(result["question_id"])
        
        logger.info(f"Already processed: {len(processed_question_ids)} questions")
        
        # Check if we can skip processing entirely
        if limit and len(existing_results) >= limit:
            logger.info(f"✓ Requested limit ({limit}) already satisfied by existing {len(existing_results)} results")
            logger.info(f"✓ No additional processing needed - skipping all LLM calls")
            logger.info(f"{'='*60}\n")
            return
        elif limit:
            logger.info(f"Existing results: {len(existing_results)}, but need {limit} total")
        
        logger.info(f"{'='*60}\n")
    
    # Load test questions
    test_file = _DATASETS_DIR / f"{dataset_name}_default_split" / "test.jsonl"
    if not test_file.exists():
        raise FileNotFoundError(f"Test file not found: {test_file}")
    
    questions = []
    with open(test_file) as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                logger.info(f"Reached limit of {limit} questions (counting from start of dataset)")
                break
            questions.append(json.loads(line))
    
    logger.info(f"Loaded {len(questions)} questions from {dataset_name} (limit: {limit or 'no limit'})")
    
    # Additional check after loading questions: if no limit, check if all questions are already processed
    if output_file.exists() and not limit:
        # Count total questions in the dataset
        total_dataset_questions = sum(1 for _ in open(test_file))
        if len(existing_results) >= total_dataset_questions:
            logger.info(f"\n{'='*60}")
            logger.info(f"✓ All {total_dataset_questions} questions in dataset already processed")
            logger.info(f"✓ No additional processing needed - skipping all LLM calls")
            logger.info(f"{'='*60}\n")
            return
    
    # Load canonical wrong answers (required)
    if not canonical_wrong_file:
        raise ValueError(
            "canonical_wrong_file is required. Run experiments with 'bare' probe first "
            "to generate canonical_wrong.jsonl"
        )
    
    canonical_wrong_path = Path(canonical_wrong_file)
    if not canonical_wrong_path.exists():
        raise FileNotFoundError(
            f"canonical_wrong.jsonl not found at: {canonical_wrong_path}\n"
            "Run experiments with 'bare' probe first to generate this file"
        )
    
    canonical_wrong = {}
    with open(canonical_wrong_path) as f:
        for i, line in enumerate(f):
            data = json.loads(line)
            canonical_wrong[i] = data["canonical_wrong"]
    logger.info(f"Loaded {len(canonical_wrong)} canonical wrong answers")
    
    # Build log filename from experiment parameters
    log_filename = f"{dataset_name}__{model_part}__i{instruction}__{cot_mode}"
    
    # Generate sentences
    generator = DatasetTierGenerator(tier1_only=tier1_only)
    
    # Update the LLM client with proper log filename if not tier1_only
    if not tier1_only and generator.llm_client:
        generator.llm_client.response_log_file = generator.log_dir / f"{log_filename}_llm_responses.jsonl"
    
    new_results = []
    failed_validations = []  # Track questions that fail validation after retries
    skipped_count = 0
    llm_call_count = 0
    retry_stats = defaultdict(int)  # Track retry statistics
    
    # Prepare questions to process
    questions_to_process = []
    missing_canonical_wrong = []
    
    # Calculate how many we need to process if limit is set
    if limit:
        remaining_needed = max(0, limit - len(existing_results))
        logger.info(f"With limit={limit} and {len(existing_results)} existing results, need {remaining_needed} more")
    else:
        remaining_needed = None
    
    for i, q in enumerate(questions):
        question_id = q["id"]
        
        # Skip if already processed
        if question_id in processed_question_ids:
            skipped_count += 1
            continue
            
        # Check canonical wrong answer exists
        if i not in canonical_wrong:
            missing_canonical_wrong.append((i, question_id))
            continue
            
        questions_to_process.append((i, q))
        
        # If we have a limit and we've collected enough questions to process, stop
        if limit and len(questions_to_process) >= remaining_needed:
            logger.info(f"Collected {len(questions_to_process)} questions to process (reaching limit of {limit} total)")
            break
    
    # If there are missing canonical wrong answers, raise an error
    if missing_canonical_wrong:
        logger.error(f"\n{'='*60}")
        logger.error(f"ERROR: Missing canonical wrong answers for {len(missing_canonical_wrong)} questions")
        logger.error(f"{'='*60}")
        for idx, qid in missing_canonical_wrong[:5]:  # Show first 5
            logger.error(f"  - Question index {idx}, ID: {qid}")
        if len(missing_canonical_wrong) > 5:
            logger.error(f"  ... and {len(missing_canonical_wrong) - 5} more")
        logger.error(f"{'='*60}")
        raise ValueError(
            f"Missing canonical wrong answers for {len(missing_canonical_wrong)} questions. "
            f"Please ensure canonical_wrong.jsonl contains entries for all {len(questions)} test questions."
        )
    
    # Calculate and log processing statistics
    total_questions = len(questions)
    already_processed = len(processed_question_ids)
    to_be_processed = len(questions_to_process)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"PROCESSING STATISTICS")
    logger.info(f"{'='*60}")
    logger.info(f"Total questions in dataset: {total_questions}")
    logger.info(f"Already processed: {already_processed}")
    logger.info(f"To be processed: {to_be_processed}")
    logger.info(f"{'='*60}\n")
    
    if not questions_to_process:
        logger.info("✓ All questions have been processed. No new LLM calls needed.")
        return
    
    # Process with progress bar
    desc = "Generating tier 1 sentences" if tier1_only else "Generating tier sentences"
    
    # For tier1-only mode, process in batch without individual updates
    if tier1_only:
        logger.info(f"Processing {len(questions_to_process)} questions in batch mode (tier 1 only)...")
        
        # Process synchronously in batch since no API calls are needed
        for idx, (i, q) in enumerate(tqdm(questions_to_process, desc=desc, disable=False)):
            question_id = q["id"]
            question_text = q["question"]
            choices = q["choices"]
            correct_letter = q["answerKey"]
            
            # Get correct answer text
            correct_idx = choices["label"].index(correct_letter)
            correct_answer = choices["text"][correct_idx]
            
            # Get wrong answer from canonical_wrong
            wrong_letter = canonical_wrong[i]
            wrong_idx = choices["label"].index(wrong_letter)
            wrong_answer = choices["text"][wrong_idx]
            
            try:
                # For tier1-only, generate directly without async since no API calls
                # Use the synchronous version by calling the method directly
                base_seed, t1_seed, t2_seed = generator._get_seeds()
                is_math_dataset = dataset_name.lower() in ['gsm8k', 'mathqa', 'aqua', 'math', 'asdiv']
                is_code_dataset = dataset_name.lower() in ['codemmlu']
                
                # Generate T1 sentences locally
                t1_user_template, t1_doc_template, t1_user_words, t1_doc_words = generator._sample_t1_elements(t1_seed, is_math_dataset, is_code_dataset)
                t1_user_correct, t1_doc_correct, t1_user_wrong, t1_doc_wrong = generator._generate_t1_sentences(
                    correct_answer, wrong_answer, t1_user_template, t1_doc_template, t1_user_words, t1_doc_words, is_code_dataset
                )
                
                # Create result with empty T2 sentences
                sentences = DatasetTierSentences(
                    question_id=question_id,
                    question=question_text,
                    correct_answer=correct_answer,
                    wrong_answer=wrong_answer,
                    t1_user_correct=t1_user_correct,
                    t1_doc_correct=t1_doc_correct,
                    t1_user_wrong=t1_user_wrong,
                    t1_doc_wrong=t1_doc_wrong,
                    t2_user_correct="",
                    t2_doc_correct="",
                    t2_user_wrong="",
                    t2_doc_wrong="",
                    model_used=generator.model,
                    temperature=generator.temperature,
                    base_seed=base_seed,
                    t1_seed=t1_seed,
                    t2_seed=t2_seed,
                    t1_user_template=t1_user_template,
                    t1_doc_template=t1_doc_template,
                    t1_user_words=t1_user_words,
                    t1_doc_words=t1_doc_words,
                    t2_user_template="",
                    t2_doc_template="",
                    t2_user_words={},
                    t2_doc_words={}
                )
                
                new_results.append(sentences.to_dict())
            except Exception as e:
                logger.error(f"Error processing question {question_id}: {e}")
                continue
    else:
        # Batch processing for full tier generation
        logger.info(f"Processing {len(questions_to_process)} questions with batch processing...")
        
        # Prepare question data for batch processing
        batch_questions = []
        for idx, (i, q) in enumerate(questions_to_process):
            question_id = q["id"]
            question_text = q["question"]
            choices = q["choices"]
            correct_letter = q["answerKey"]
            
            # Get correct answer text
            correct_idx = choices["label"].index(correct_letter)
            correct_answer = choices["text"][correct_idx]
            
            # Get wrong answer from canonical_wrong
            wrong_letter = canonical_wrong[i]
            wrong_idx = choices["label"].index(wrong_letter)
            wrong_answer = choices["text"][wrong_idx]
            
            # Skip if already processed
            if question_id in processed_question_ids:
                skipped_count += 1
                continue
            
            # Apply lowercase for CodeMMLU answers
            is_code_dataset = dataset_name.lower() in ['codemmlu']
            if is_code_dataset:
                correct_answer = correct_answer.lower()
                wrong_answer = wrong_answer.lower()
            
            batch_questions.append({
                "idx": i,
                "question_id": question_id,
                "question": question_text,
                "correct_answer": correct_answer,
                "wrong_answer": wrong_answer,
                "is_math_dataset": dataset_name.lower() in ['gsm8k', 'mathqa', 'aqua', 'math', 'asdiv'],
                "is_code_dataset": is_code_dataset
            })
        
        # Process in batches
        
        for batch_start in range(0, len(batch_questions), batch_size):
            batch_end = min(batch_start + batch_size, len(batch_questions))
            current_batch = batch_questions[batch_start:batch_end]
            
            logger.info(f"Processing batch {batch_start // batch_size + 1}/{(len(batch_questions) + batch_size - 1) // batch_size} "
                       f"(questions {batch_start + 1}-{batch_end} of {len(batch_questions)})")
            
            # Progress tracking for current batch
            progress_count = 0
            def progress_callback(completed, total):
                nonlocal progress_count
                progress_count = completed
            
            # Process batch
            batch_results = await generator.generate_batch_tiers(
                current_batch,
                max_concurrent=max_concurrent,
                progress_callback=progress_callback
            )
            
            # Process results with validation
            successful_results = []
            failed_to_retry = []
            
            for result in batch_results:
                metadata = result.get("metadata", {})
                
                if result.get("error"):
                    logger.error(f"Failed on question {metadata.get('question_id')}: {result['error']}")
                    continue
                
                sentences = result["sentences"]
                if not sentences:
                    continue
                
                # Validate T2 sentences (skip if tier1_only)
                validation_errors = {}
                if not generator.tier1_only:
                    validation_errors = generator._validate_t2_sentences(sentences)
                
                if validation_errors:
                    # Add to retry list
                    failed_to_retry.append({
                        "result": result,
                        "attempt": 1,
                        "validation_errors": validation_errors
                    })
                    logger.warning(f"Question {metadata['question_id']}: T2 validation failed, will retry")
                else:
                    successful_results.append(result)
                    llm_call_count += 1
            
            # Retry failed validations up to 5 times per question
            max_retries_per_question = 5
            while failed_to_retry:
                logger.info(f"Retrying {len(failed_to_retry)} questions with validation errors...")
                
                # Prepare retry batch
                retry_batch = []
                for item in failed_to_retry:
                    metadata = item["result"]["metadata"]
                    retry_batch.append({
                        "idx": metadata["idx"],
                        "question_id": metadata["question_id"],
                        "question": metadata["question"],
                        "correct_answer": metadata["correct_answer"],
                        "wrong_answer": metadata["wrong_answer"],
                        "is_math_dataset": metadata.get("is_math_dataset", False),
                        "is_code_dataset": metadata.get("is_code_dataset", False),
                        "_retry_attempt": item["attempt"]
                    })
                
                # Process retries
                retry_results = await generator.generate_batch_tiers(
                    retry_batch,
                    max_concurrent=max_concurrent,
                    progress_callback=progress_callback
                )
                
                # Check retry results
                still_failed = []
                for i, retry_result in enumerate(retry_results):
                    original_item = failed_to_retry[i]
                    retry_metadata = retry_result.get("metadata", {})
                    
                    if retry_result.get("error"):
                        logger.error(f"Retry failed for {retry_metadata.get('question_id')}: {retry_result['error']}")
                        if original_item["attempt"] < max_retries_per_question:
                            original_item["attempt"] += 1
                            still_failed.append(original_item)
                        else:
                            # Max retries reached, save as failed
                            failed_validations.append({
                                "idx": retry_metadata["idx"],
                                "question_id": retry_metadata["question_id"],
                                "question": retry_metadata["question"],
                                "correct_answer": retry_metadata["correct_answer"],
                                "wrong_answer": retry_metadata["wrong_answer"],
                                "validation_errors": {"error": "Max retries reached"},
                                "sentences": None
                            })
                        continue
                    
                    sentences = retry_result["sentences"]
                    if not sentences:
                        continue
                    
                    # Validate again
                    validation_errors = {}
                    if not generator.tier1_only:
                        validation_errors = generator._validate_t2_sentences(sentences)
                    
                    if validation_errors:
                        if original_item["attempt"] < max_retries_per_question:
                            original_item["attempt"] += 1
                            original_item["validation_errors"] = validation_errors
                            still_failed.append(original_item)
                            logger.warning(f"Question {retry_metadata['question_id']}: Still failing validation on attempt {original_item['attempt']}")
                        else:
                            # Max retries reached, save as failed
                            logger.error(f"Question {retry_metadata['question_id']}: Failed validation after {max_retries_per_question} attempts")
                            failed_validations.append({
                                "idx": retry_metadata["idx"],
                                "question_id": retry_metadata["question_id"],
                                "question": retry_metadata["question"],
                                "correct_answer": retry_metadata["correct_answer"],
                                "wrong_answer": retry_metadata["wrong_answer"],
                                "validation_errors": validation_errors,
                                "sentences": sentences.to_dict(),
                                "attempts": max_retries_per_question
                            })
                    else:
                        # Success!
                        successful_results.append(retry_result)
                        llm_call_count += original_item["attempt"] + 1
                        retry_stats[original_item["attempt"] + 1] += 1
                        logger.info(f"Question {retry_metadata['question_id']}: Passed validation on attempt {original_item['attempt'] + 1}")
                
                failed_to_retry = still_failed
            
            # Add all successful results to new_results
            for result in successful_results:
                metadata = result.get("metadata", {})
                sentences = result["sentences"]
                
                new_results.append({
                    "idx": metadata["idx"],
                    "question_id": metadata["question_id"],
                    "sentences": sentences.to_dict(),
                    "validation_passed": True
                })
            
            logger.info(f"Completed batch {batch_start // batch_size + 1} - processed {len(successful_results)} questions successfully")
    
    # Combine existing and new results
    all_results = existing_results + new_results
    
    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Extract just the sentences data (without wrapper) for the final output
    sentences_only = []
    for result in all_results:
        if isinstance(result, dict):
            if "sentences" in result:
                # New format with wrapper - extract sentences
                sentences_only.append(result["sentences"])
            else:
                # Old format or tier1-only format - use as is
                sentences_only.append(result)
    
    with open(output_file, 'w') as f:
        json.dump(sentences_only, f, indent=2)
    
    # Save failed validations if any
    if failed_validations:
        failed_file = output_file.parent / f"{output_file.stem}_failed_validations.json"
        with open(failed_file, 'w') as f:
            json.dump(failed_validations, f, indent=2)
        logger.warning(f"\n⚠️  {len(failed_validations)} questions failed validation after retries")
        logger.warning(f"Failed validations saved to: {failed_file}")
    
    logger.info(f"\n{'='*60}")
    logger.info(f"FINAL SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"  LLM calls made: {llm_call_count}")
    logger.info(f"  LLM calls skipped: {skipped_count}")
    logger.info(f"  New results generated: {len(new_results)}")
    logger.info(f"  Questions passed validation: {len(new_results) - len(failed_validations)}")
    logger.info(f"  Questions failed validation: {len(failed_validations)}")
    logger.info(f"  Total questions in file: {len(all_results)}")
    logger.info(f"  Saved to: {output_file}")
    
    if retry_stats:
        logger.info(f"\n  Retry Statistics:")
        for attempts, count in sorted(retry_stats.items()):
            logger.info(f"    {count} questions succeeded on attempt {attempts}")
    
    logger.info(f"{'='*60}")


def main():
    import argparse
    import yaml
    import sys
    from pathlib import Path
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s'
    )
    
    # Suppress HTTP request logs from OpenAI and urllib3
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    
    # Add project root to path
    ROOT = Path(__file__).resolve().parents[1]
    sys.path.append(str(ROOT))
    
    from core.exp_name import parse_experiment_name
    
    parser = argparse.ArgumentParser(description="Generate tier sentences for datasets")
    parser.add_argument("--config", required=True, help="Path to experiment config file")
    parser.add_argument("--experiment", help="Process only this specific experiment")
    parser.add_argument("--limit", type=int, help="Limit number of questions to process (for testing)")
    parser.add_argument("--results-root", default="results", help="Root directory for results (default: results)")
    parser.add_argument("--tier1-only", action="store_true", help="Only generate tier 1 sentences (for testing)")
    parser.add_argument("--batch-size", type=int, default=50, help="Number of questions to process per batch (default: 50)")
    parser.add_argument("--max-concurrent", type=int, default=20, help="Maximum concurrent API calls per batch (default: 20)")
    args = parser.parse_args()
    
    # Load config
    cfg = yaml.safe_load(Path(args.config).read_text())
    experiments = cfg.get("experiments", [])
    
    if not experiments:
        sys.exit("No experiments found in config.")
    
    # Filter by experiment name if provided
    if args.experiment:
        experiments = [exp for exp in experiments if exp.get("name") == args.experiment]
        if not experiments:
            sys.exit(f"Experiment '{args.experiment}' not found in config")
        logger.info(f"Processing only experiment: {args.experiment}")
    
    # Process each unique dataset from experiments
    datasets_processed = set()
    
    for exp_config in experiments:
        experiment_name = exp_config.get("name")
        if not experiment_name:
            logger.warning("Experiment missing 'name' field, skipping")
            continue
            
        # Parse experiment to get dataset and model
        try:
            exp = parse_experiment_name(experiment_name)
            dataset = exp.dataset
            model_key = exp.model_key
            instruction = exp.instruction
            cot_mode = "cot" if exp.use_cot else "nocot"
            reasoning_mode = exp.reasoning_mode
            
            # Build filename similar to exp_name structure
            model_part = model_key
            if reasoning_mode:
                model_part = model_key + "r"
            
            # Create unique identifier for tracking
            dataset_model_variant = f"{dataset}__{model_part}__i{instruction}__{cot_mode}"
            
            if dataset_model_variant in datasets_processed:
                logger.warning(f"Dataset-model-variant '{dataset_model_variant}' already encountered in config - skipping duplicate experiment '{experiment_name}'")
                continue
                
            # Check if output file already exists and might be complete
            output_file = Path(_DEFAULT_TIER_DIR) / f"{dataset}__{model_part}__i{instruction}__{cot_mode}_tier_sentences.json"
            if output_file.exists():
                # Check how many questions are already processed
                with open(output_file, 'r') as f:
                    existing_data = json.load(f)
                    existing_count = len(existing_data)
                
                # Handle both limit and no-limit cases
                if args.limit:
                    if existing_count >= args.limit:
                        logger.info(f"\n{'='*60}")
                        logger.info(f"Dataset-model-variant '{dataset_model_variant}' already has {existing_count} results")
                        logger.info(f"Requested limit ({args.limit}) already satisfied - skipping")
                        logger.info(f"{'='*60}\n")
                        datasets_processed.add(dataset_model_variant)
                        continue
                else:
                    # Count total questions in test file
                    test_file = _DATASETS_DIR / f"{dataset}_default_split" / "test.jsonl"
                    if test_file.exists():
                        total_count = sum(1 for _ in open(test_file))
                        if existing_count >= total_count:
                            logger.info(f"\n{'='*60}")
                            logger.info(f"Dataset-model-variant '{dataset_model_variant}' already fully processed")
                            logger.info(f"All {existing_count} questions processed - skipping")
                            logger.info(f"{'='*60}\n")
                            datasets_processed.add(dataset_model_variant)
                            continue
            
            # Look for canonical_wrong.jsonl in the experiment's results directory
            canonical_wrong_path = Path(args.results_root) / experiment_name / "canonical_wrong.jsonl"
            
            if not canonical_wrong_path.exists():
                logger.warning(f"canonical_wrong.jsonl not found for {experiment_name} at {canonical_wrong_path}")
                logger.warning("Run experiments with 'bare' probe first to generate canonical wrong answers")
                continue
            
            logger.info(f"\nProcessing dataset-model-variant: {dataset_model_variant} from experiment: {experiment_name}")
            asyncio.run(generate_dataset_tiers(
                dataset, 
                model_key, 
                instruction,
                cot_mode,
                reasoning_mode,
                str(canonical_wrong_path), 
                limit=args.limit,
                tier1_only=args.tier1_only,
                batch_size=args.batch_size,
                max_concurrent=args.max_concurrent
            ))
            datasets_processed.add(dataset_model_variant)
            
        except Exception as e:
            import traceback
            logger.error(f"Error processing experiment {experiment_name}: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            continue
    
    if not datasets_processed:
        logger.error("No dataset-model-variants were processed. Make sure experiments have been run with 'bare' probe first.")
    else:
        logger.info(f"Processed {len(datasets_processed)} dataset-model-variant(s): {', '.join(datasets_processed)}")


if __name__ == "__main__":
    main()