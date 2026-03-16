"""Prompt templates for RLM sessions."""

from __future__ import annotations

REPL_INTRO = """\
You have access to a Python REPL with a large context pre-loaded.
The context is available as the variable CONTEXT (a string).

Available helper functions:
- peek(start=0, length=500) -> str: preview a slice of the context
- grep(pattern, context_lines=0, max_results=50) -> list[str]: regex search
- search(query, max_results=20) -> list[str]: keyword search
- chunk(chunk_size=4000, overlap=200) -> list[str]: split into chunks
- count_lines() -> int: number of lines in the context

Context metadata variables:
- CONTEXT_LENGTH: total character count
- CONTEXT_TYPE: "str", "list", "dict", or "file"
- CONTEXT_SIZE_TOKENS: estimated token count
"""

FINAL_INSTRUCTIONS = """\
When you have the answer, signal it by calling:
  FINAL("your answer here")
or, if the answer is stored in a variable:
  FINAL_VAR("variable_name")

You MUST call FINAL or FINAL_VAR to complete the task.
"""

SUB_QUERY_INSTRUCTIONS = """\
You can spawn a sub-query on a subset of the context:
  result = rlm_sub_query(query="sub question", context="subset of context")
This runs a nested RLM session and returns the result.
"""

BUDGET_WARNING = """\
WARNING: You have {remaining} iterations left. Please produce your final answer soon.
If you do not call FINAL() within the remaining iterations, a forced FINAL will
be generated from whatever partial results are available.
"""

ROOT_TEMPLATE = """\
{repl_intro}

{final_instructions}

{sub_query_section}

## Task
{query}

## Context Info
- Type: {context_type}
- Length: {context_length} characters (~{context_tokens} tokens)
- Lines: {num_lines}

Write Python code in ```python ... ``` blocks to explore the context and answer the question.
"""

SUB_TEMPLATE = """\
{repl_intro}

{final_instructions}

## Sub-Task (depth {depth})
{query}

## Context Info
- Type: {context_type}
- Length: {context_length} characters (~{context_tokens} tokens)

Write Python code to answer the sub-question using the provided context slice.
"""
