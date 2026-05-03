"""
LLM-based semantic similarity scorer for OmniGAIA.

Uses an LLM to judge whether two answers are semantically equivalent,
handling format differences like "January 2018" vs "2018-01".
"""

from __future__ import annotations

import asyncio
import os
from typing import Optional, Tuple

from openai import AsyncOpenAI

from base.engine.async_llm import LLMsConfig, llm_retry
from base.engine.logs import logger


# Default model for semantic scoring
DEFAULT_JUDGE_MODEL = "gpt-4o"

# Prompt for semantic similarity judgment
JUDGE_PROMPT = """You are a precise answer evaluator. Given the original question, determine if the predicted answer is semantically equivalent to the expected answer.

IMPORTANT RULES:
1. Focus on SEMANTIC MEANING, not exact string match.
2. Date format differences are EQUIVALENT: "January 2018" ~ "2018-01" ~ "Jan 2018".
3. Currency format differences are EQUIVALENT: "$500 billion" ~ "500 billion dollars".
4. Percentage format differences are EQUIVALENT: "90%" ~ "90 percent".
5. Approximate values are EQUIVALENT: "nearly 90%" ~ "90%", "about 500" ~ "500".
6. A numeric value WITH or WITHOUT a unit/label word is EQUIVALENT as long as the number is the same:
   - "102,924 times" ~ "102,924" (same number, extra unit word is OK)
   - "3.5 million" ~ "3,500,000" (same value in different notation)
   - "42 years" ~ "42" (same number, extra unit word is OK)
7. Thousand separators and formatting differences are EQUIVALENT: "102,924" ~ "102924".
8. Numbers must actually match in value: "50" and "500" are NOT equivalent.
9. If the predicted answer contains the expected answer with correct context, it's EQUIVALENT. This applies regardless of the separator or format used:
   - "K2; Marker" contains "Marker" → EQUIVALENT to "Marker"
   - "Brand: K2, Binding: Marker" contains "Marker" → EQUIVALENT to "Marker"
   - "The answer is Marker (K2's partner)" contains "Marker" → EQUIVALENT to "Marker"
10. **LIST / MULTI-PART ANSWER TOLERANCE**: If the predicted answer is a list or multi-part response (separated by semicolons, commas, slashes, newlines, "and", or other delimiters), and the expected answer matches ONE of the items in that list, treat it as EQUIVALENT. The predicted answer may include additional correct context (e.g., intermediate reasoning results) alongside the final answer — this is acceptable as long as the expected answer is clearly present.
   - "K2; Marker" with expected "Marker" → EQUIVALENT (Marker is one of the listed items)
   - "Paris; France" with expected "Paris" → EQUIVALENT
   - "42 / forty-two" with expected "42" → EQUIVALENT
11. Order of list items matters ONLY when the question explicitly asks for a specific ordering. Otherwise, "A, B" and "B, A" are EQUIVALENT.
12. Minor wording differences that preserve meaning are EQUIVALENT: "the United States" ~ "United States" ~ "US" ~ "USA".
13. Use the original question to understand what is being asked, so you can better judge whether the predicted answer correctly addresses the question and matches the expected answer.
14. **EXTRA CONTEXT IS OK**: If the predicted answer includes extra information beyond the expected answer (e.g., explanations, labels, intermediate steps), but the core answer matches, it is still EQUIVALENT. Focus on whether the KEY ANSWER is correct, not whether there is extra text.

Original Question: {question}
Expected Answer: {expected}
Predicted Answer: {predicted}

Respond with ONLY one word: "EQUIVALENT" or "DIFFERENT"
"""


def _get_llm_config(model_name: Optional[str] = None) -> Tuple[str, str, str]:
    """Get LLM configuration from model config or environment variables."""
    model_name = model_name or DEFAULT_JUDGE_MODEL
    
    # Try to get config from LLMsConfig
    try:
        llms_config = LLMsConfig.default()
        model_config = llms_config.get(model_name)
        if model_config:
            return (
                model_config.key,
                model_config.base_url,
                model_config.model,
            )
    except Exception:
        pass
    
    # Fallback to environment variables
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("LLM_API_BASE") or "https://api.openai.com/v1"
    
    return (api_key, base_url, model_name)


async def llm_semantic_score(
    predicted: str,
    expected: str,
    model: Optional[str] = None,
    question: Optional[str] = None,
) -> float:
    """
    Use LLM to judge semantic equivalence between predicted and expected answers.
    
    Args:
        predicted: The model's predicted answer
        expected: The expected correct answer
        model: Optional model name to use for judging
        question: Optional original question for context
        
    Returns:
        1.0 if semantically equivalent, 0.0 otherwise
    """
    if not predicted or not expected:
        return 0.0
    
    # Quick check: exact match (normalized)
    if predicted.strip().lower() == expected.strip().lower():
        return 1.0
    
    api_key, base_url, model_name = _get_llm_config(model)
    
    if not api_key:
        logger.warning("[LLM Scorer] No API key available, falling back to 0.0")
        return 0.0
    
    client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    
    prompt = JUDGE_PROMPT.format(
        question=question or "(not provided)",
        expected=expected,
        predicted=predicted,
    )
    
    try:
        completion = await llm_retry(
            client.chat.completions.create,
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=10,
            label=f"LLMScorer({model_name})",
        )
        
        response = completion.choices[0].message.content.strip().upper()
        
        if "EQUIVALENT" in response:
            logger.info(f"[LLM Scorer] EQUIVALENT: '{predicted}' ~ '{expected}'")
            return 1.0
        else:
            logger.info(f"[LLM Scorer] DIFFERENT: '{predicted}' ≠ '{expected}'")
            return 0.0
            
    except Exception as e:
        logger.error(f"[LLM Scorer] Error: {e}")
        return 0.0


def llm_semantic_score_sync(
    predicted: str,
    expected: str,
    model: Optional[str] = None,
    question: Optional[str] = None,
) -> float:
    """Synchronous wrapper for llm_semantic_score."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If we're in an async context, create a new task
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    asyncio.run,
                    llm_semantic_score(predicted, expected, model, question)
                )
                return future.result(timeout=30)
        else:
            return loop.run_until_complete(
                llm_semantic_score(predicted, expected, model, question)
            )
    except Exception as e:
        logger.error(f"[LLM Scorer Sync] Error: {e}")
        return 0.0
