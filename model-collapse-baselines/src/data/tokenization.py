"""Shared tokenizer utilities."""

from __future__ import annotations


def get_tokenizer(model_name: str):
    """Load and configure a tokenizer with proper padding settings.

    Args:
        model_name: HuggingFace model name or local path.

    Returns:
        A configured AutoTokenizer instance.
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Ensure pad token is set -- many causal LM tokenizers lack one.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Left-padding is preferred for batched generation with causal LMs.
    tokenizer.padding_side = "left"

    return tokenizer
