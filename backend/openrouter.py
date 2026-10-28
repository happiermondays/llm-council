"""Anthropic API client for making LLM requests.

(Filename kept as openrouter.py so existing imports keep working; the app now
talks directly to the Anthropic Messages API instead of OpenRouter.)
"""

import asyncio
import httpx
from typing import List, Dict, Any, Optional
from .config import (
    ANTHROPIC_API_KEY,
    ANTHROPIC_API_URL,
    ANTHROPIC_VERSION,
    MAX_TOKENS,
)


async def query_model(
    model: str,
    messages: List[Dict[str, str]],
    timeout: float = 120.0
) -> Optional[Dict[str, Any]]:
    """
    Query a single Claude model via the Anthropic Messages API.

    Args:
        model: Anthropic model identifier (e.g., "claude-sonnet-5-5")
        messages: List of message dicts with 'role' and 'content'
        timeout: Request timeout in seconds

    Returns:
        Response dict with 'content' and 'reasoning_details' (always None here,
        kept for compatibility with the rest of the app), or None if failed
    """
    headers = {
        "x-api-key": ANTHROPIC_API_KEY or "",
        "anthropic-version": ANTHROPIC_VERSION,
        "content-type": "application/json",
    }

    payload = {
        "model": model,
        "max_tokens": MAX_TOKENS,
        "messages": messages,
    }

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                ANTHROPIC_API_URL,
                headers=headers,
                json=payload
            )
            response.raise_for_status()

            data = response.json()
            # Response content is a list of blocks; join the text blocks
            text = "".join(
                block.get("text", "")
                for block in data.get("content", [])
                if block.get("type") == "text"
            )

            return {
                'content': text,
                'reasoning_details': None
            }

    except Exception as e:
        print(f"Error querying model {model}: {e}")
        return None


async def query_models_parallel(
    models: List[str],
    messages: List[Dict[str, str]]
) -> Dict[str, Optional[Dict[str, Any]]]:
    """
    Query multiple models in parallel.

    Args:
        models: List of Anthropic model identifiers
        messages: List of message dicts to send to each model

    Returns:
        Dict mapping model identifier to response dict (or None if failed)
    """
    # Create tasks for all models
    tasks = [query_model(model, messages) for model in models]

    # Wait for all to complete
    responses = await asyncio.gather(*tasks)

    # Map models to their responses
    return {model: response for model, response in zip(models, responses)}
