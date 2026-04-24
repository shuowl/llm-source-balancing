"""
LLM Client Module
================

This module provides a robust, async-first client for interacting with OpenAI's API,
specifically designed for high-throughput batch processing with proper error handling,
logging, and retry mechanisms.
"""

import json
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging
from openai import AsyncOpenAI
import os

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_LLM_LOG_DIR = _PROJECT_ROOT / "logs" / "llm_responses"

logger = logging.getLogger(__name__)


class LLMClient:
    """
    Asynchronous LLM client with comprehensive logging and retry capabilities.
    
    This client provides a robust interface to OpenAI's API with features designed
    for production use cases requiring high reliability and observability.
    
    Features:
        - Automatic retry with exponential backoff for transient failures
        - Comprehensive logging of all API interactions to JSONL files
        - Async-first design for high-throughput batch processing
        - Configurable temperature and model selection
        - Custom log file naming for experiment tracking
    
    Attributes:
        api_key (str): OpenAI API key
        client (AsyncOpenAI): Async OpenAI client instance
        model (str): Model to use (default: gpt-4o)
        temperature (float): Generation temperature (default: 0.3)
        max_retries (int): Maximum retry attempts (default: 3)
        log_dir (Path): Directory for log files
        response_log_file (Path): Path to current log file
    
    Example:
        >>> client = LLMClient()
        >>> response = await client.generate("What is 2+2?")
        >>> print(response)
        "4"
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o",
        temperature: float = 0.3,
        max_retries: int = 5,
        log_dir: Optional[str] = None,
        log_filename: Optional[str] = None
    ):
        """
        Initialize LLM client
        
        Args:
            api_key: OpenAI API key (defaults to env var)
            model: Model to use
            temperature: Temperature for generation
            max_retries: Maximum retries for failed requests
            log_dir: Directory to log responses
            log_filename: Custom log filename (without extension)
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY env var.")
        
        self.client = AsyncOpenAI(api_key=self.api_key)
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        
        # Set up logging
        if log_dir:
            self.log_dir = Path(log_dir)
        else:
            self.log_dir = _DEFAULT_LLM_LOG_DIR
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create log file with custom name or timestamp
        if log_filename:
            self.response_log_file = self.log_dir / f"{log_filename}_llm_responses.jsonl"
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.response_log_file = self.log_dir / f"llm_responses_{timestamp}.jsonl"
    
    def _log_response(
        self,
        prompt: str,
        response: str,
        metadata: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None
    ):
        """
        Log LLM response to file immediately after generation.
        
        Creates a JSONL entry with timestamp, prompt, response, and any metadata.
        Each line is a complete JSON object for easy streaming and processing.
        
        Args:
            prompt: The input prompt sent to the LLM
            response: The LLM's response (empty string if error)
            metadata: Optional dict with additional context (e.g., question_id, attempt)
            error: Optional error message if the request failed
        
        Log Format:
            {
                "timestamp": "2024-01-01T12:00:00",
                "model": "gpt-4o",
                "temperature": 0.3,
                "prompt": "...",
                "response": "...",
                "error": null,
                "metadata": {...}
            }
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "model": self.model,
            "temperature": self.temperature,
            "prompt": prompt,
            "response": response,
            "error": error,
            "metadata": metadata or {}
        }
        
        with open(self.response_log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    async def generate(
        self,
        prompt: str,
        max_tokens: int = 400,
        metadata: Optional[Dict[str, Any]] = None,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Generate text from prompt with retries and logging
        
        Args:
            prompt: User prompt
            max_tokens: Maximum tokens to generate
            metadata: Additional metadata to log
            system_prompt: Optional system prompt
            
        Returns:
            Generated text
            
        Raises:
            Exception: If all retries fail
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        for attempt in range(1, self.max_retries + 1):
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=max_tokens
                )
                
                content = response.choices[0].message.content.strip()
                
                # Log successful response
                self._log_response(
                    prompt=prompt,
                    response=content,
                    metadata={
                        **(metadata or {}),
                        "attempt": attempt,
                        "max_tokens": max_tokens,
                        "system_prompt": system_prompt
                    }
                )
                
                return content
                
            except Exception as e:
                error_msg = f"Attempt {attempt}/{self.max_retries} failed: {str(e)}"
                logger.warning(error_msg)
                
                # Log failed attempt
                self._log_response(
                    prompt=prompt,
                    response="",
                    metadata={
                        **(metadata or {}),
                        "attempt": attempt,
                        "max_tokens": max_tokens
                    },
                    error=str(e)
                )
                
                if attempt < self.max_retries:
                    # Exponential backoff
                    wait_time = 2 ** (attempt - 1)
                    logger.info(f"Waiting {wait_time}s before retry...")
                    await asyncio.sleep(wait_time)
                else:
                    raise Exception(f"All {self.max_retries} attempts failed for prompt") from e
    
    async def generate_batch(
        self,
        prompts: List[Dict[str, Any]],
        max_concurrent: int = 10,
        progress_callback: Optional[callable] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate responses for multiple prompts concurrently with controlled parallelism.
        
        This method processes multiple prompts efficiently using asyncio.Semaphore to
        limit concurrent API calls, preventing rate limit issues while maximizing throughput.
        
        Args:
            prompts: List of prompt dictionaries, each containing:
                - prompt (str): The text prompt (required)
                - metadata (dict): Additional context to log (optional)
                - max_tokens (int): Max tokens to generate (default: 400)
                - system_prompt (str): System message (optional)
            max_concurrent: Maximum number of concurrent API calls (default: 10)
            progress_callback: Optional callback function(completed, total) for progress updates
            
        Returns:
            List of result dictionaries, each containing:
                - index (int): Original position in input list
                - prompt (str): The input prompt
                - response (str): Generated text (None if error)
                - error (str): Error message (None if successful)
                - metadata (dict): Original metadata passed in
        
        Example:
            >>> prompts = [
            ...     {"prompt": "What is 2+2?", "metadata": {"id": 1}},
            ...     {"prompt": "What is 3+3?", "metadata": {"id": 2}}
            ... ]
            >>> results = await client.generate_batch(prompts, max_concurrent=20)
            >>> for r in results:
            ...     print(f"Q: {r['prompt']} A: {r['response']}")
        
        Note:
            Results are returned in the same order as input prompts, regardless of
            completion order. All logging happens immediately as responses arrive.
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_single(prompt_data: Dict[str, Any], index: int) -> Dict[str, Any]:
            async with semaphore:
                try:
                    response = await self.generate(
                        prompt=prompt_data["prompt"],
                        max_tokens=prompt_data.get("max_tokens", 400),
                        metadata={
                            **prompt_data.get("metadata", {}),
                            "batch_index": index
                        },
                        system_prompt=prompt_data.get("system_prompt")
                    )
                    
                    result = {
                        "index": index,
                        "prompt": prompt_data["prompt"],
                        "response": response,
                        "error": None,
                        "metadata": prompt_data.get("metadata", {})
                    }
                    
                except Exception as e:
                    result = {
                        "index": index,
                        "prompt": prompt_data["prompt"],
                        "response": None,
                        "error": str(e),
                        "metadata": prompt_data.get("metadata", {})
                    }
                
                if progress_callback:
                    progress_callback(index, len(prompts))
                
                return result
        
        # Process all prompts concurrently
        tasks = [
            process_single(prompt_data, i)
            for i, prompt_data in enumerate(prompts)
        ]
        
        results = await asyncio.gather(*tasks)
        
        # Sort by original index
        results.sort(key=lambda x: x["index"])
        
        return results


class TierSentenceGenerator:
    """Specialized generator for tier sentences using LLM"""
    
    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client
    
    async def generate_t2_sentences(
        self,
        question: str,
        correct_answer: str,
        wrong_answer: str,
        user_template: str,
        doc_template: str,
        user_words: Dict[str, str],
        doc_words: Dict[str, str],
        is_math_dataset: bool = False,
        is_code_dataset: bool = False,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, str]:
        """
        Generate T2 contextual sentences using LLM
        
        Returns:
            Dict with t2_user_correct, t2_doc_correct, t2_user_wrong, t2_doc_wrong
        """
        prompt = self._build_t2_prompt(
            question, correct_answer, wrong_answer,
            user_template, doc_template, user_words, doc_words,
            is_math_dataset, is_code_dataset
        )
        
        response = await self.llm.generate(
            prompt=prompt,
            max_tokens=400,
            metadata=metadata
        )
        
        return self._parse_t2_response(response)
    
    def _build_t2_prompt(
        self,
        question: str,
        correct_answer: str,
        wrong_answer: str,
        user_template: str,
        doc_template: str,
        user_words: Dict[str, str],
        doc_words: Dict[str, str],
        is_math_dataset: bool,
        is_code_dataset: bool = False
    ) -> str:
        """Build the T2 generation prompt"""
        example = self._get_dataset_specific_example(is_math_dataset, is_code_dataset, correct_answer, wrong_answer)
        
        return f"""Generate contextual sentences for this multiple choice question.

Question: {question}
Correct answer: {correct_answer}
Wrong answer: {wrong_answer}

You must use EXACTLY these pre-selected templates and words:

User template: {user_template}
User words: {user_words}

Doc template: {doc_template}
Doc words: {doc_words}

YOUR TASK:
1. Create ONE contextual claim that incorporates key words from the question
2. This contextual claim will contain a placeholder [answer] where the answer goes
3. Use this EXACT SAME contextual claim in ALL 4 sentences - DO NOT VARY IT AT ALL

CRITICAL REQUIREMENT - THE CONTEXTUAL CLAIM MUST BE IDENTICAL:
- First, write your contextual claim with [answer] as placeholder
- Then use this EXACT claim in all 4 sentences, only changing [answer] to the correct/wrong answer
- The ONLY difference between sentences should be:
  a) The template/word variations (user vs doc)
  b) The answer word (correct vs wrong)
- Everything else, including the contextual claim, must be EXACTLY IDENTICAL

{example}

Generate the 4 sentences by:
1. Create your contextual claim with [answer] placeholder
2. Fill the templates with the provided words
3. Replace {{contextual_claim}} with your contextual claim
4. IMPORTANT: Replace [answer] with the ACTUAL answer value:
   - In USER_CORRECT and DOC_CORRECT: Replace [answer] with {correct_answer}
   - In USER_WRONG and DOC_WRONG: Replace [answer] with {wrong_answer}
5. DO NOT leave "[answer]" in any final sentence - it must be replaced!

Format your response EXACTLY as:
USER_CORRECT: [sentence]
DOC_CORRECT: [sentence]
USER_WRONG: [sentence]
DOC_WRONG: [sentence]

IMPORTANT: Do NOT add quotation marks around the sentences. Output plain text only."""
    
    def _get_dataset_specific_example(self, is_math_dataset: bool, is_code_dataset: bool, correct_answer: str, wrong_answer: str) -> str:
        """Get dataset-specific example for the prompt"""
        if is_math_dataset:
            return f"""EXAMPLE to illustrate the requirement:
If your contextual claim is "Janet sells [answer] eggs at the farmers' market daily"
Then you MUST REPLACE [answer] with the actual answer values:
- USER_CORRECT: "I've calculated Janet sells {correct_answer} eggs at the farmers' market daily."
- DOC_CORRECT: "Solutions show Janet sells {correct_answer} eggs at the farmers' market daily."
- USER_WRONG: "I've calculated Janet sells {wrong_answer} eggs at the farmers' market daily."
- DOC_WRONG: "Solutions show Janet sells {wrong_answer} eggs at the farmers' market daily."

CRITICAL: DO NOT leave "[answer]" in your final sentences! Replace it with the actual values."""
        elif is_code_dataset:
            return f"""EXAMPLE to illustrate the requirement:
If your contextual claim is "the output of this code segment is [answer]"
Then you MUST REPLACE [answer] with the actual answer values:
- USER_CORRECT: "I've tested the output of this code segment is {correct_answer}."
- DOC_CORRECT: "Technical references indicate the output of this code segment is {correct_answer}."
- USER_WRONG: "I've tested the output of this code segment is {wrong_answer}."
- DOC_WRONG: "Technical references indicate the output of this code segment is {wrong_answer}."

CRITICAL: DO NOT leave "[answer]" in your final sentences! Replace it with the actual values."""
        else:
            return f"""EXAMPLE to illustrate the requirement:
If your contextual claim is "revolving doors serve as security measures at a [answer]"
Then you MUST REPLACE [answer] with the actual answer values:
- USER_CORRECT: "I've seen revolving doors serve as security measures at a {correct_answer}."
- DOC_CORRECT: "Revolving doors serve as security measures at a {correct_answer}, according to available sources."
- USER_WRONG: "I've seen revolving doors serve as security measures at a {wrong_answer}."
- DOC_WRONG: "Revolving doors serve as security measures at a {wrong_answer}, according to available sources."

CRITICAL: DO NOT leave "[answer]" in your final sentences! Replace it with the actual values."""
    
    def _parse_t2_response(self, response: str) -> Dict[str, str]:
        """Parse the GPT response for T2 sentences"""
        sentences = {}
        
        for line in response.strip().split('\n'):
            line = line.strip()
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip().lower()
                value = value.strip()
                
                # Map to expected fields
                if key == "user_correct":
                    sentences["t2_user_correct"] = value
                elif key == "doc_correct":
                    sentences["t2_doc_correct"] = value
                elif key == "user_wrong":
                    sentences["t2_user_wrong"] = value
                elif key == "doc_wrong":
                    sentences["t2_doc_wrong"] = value
        
        # Validate we got all T2 fields
        required = ["t2_user_correct", "t2_doc_correct", "t2_user_wrong", "t2_doc_wrong"]
        missing = [f for f in required if f not in sentences]
        if missing:
            raise ValueError(f"Missing T2 fields in response: {missing}\nResponse:\n{response}")
        
        return sentences