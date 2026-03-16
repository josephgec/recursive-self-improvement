"""Prompt templates for LLM-powered genetic operators."""

INIT_DIRECT = """Write a Python function `transform(input_grid)` that transforms the input grid to produce the output grid.

{task_description}

Return ONLY the Python function, no explanations.
"""

INIT_PATTERN = """Analyze the pattern in these ARC examples and write a `transform(input_grid)` function.

{task_description}

Think step by step about the pattern, then write the function.
"""

INIT_DECOMPOSE = """Break down this ARC task into sub-problems and solve each one.

{task_description}

Write a `transform(input_grid)` function that handles all sub-problems.
"""

INIT_ANALOGY = """This ARC task is similar to common grid transformations (rotation, reflection, color mapping, etc.).

{task_description}

Identify the closest analogy and write a `transform(input_grid)` function.
"""

INIT_CONSTRAINT = """Given the constraints shown in the examples, write a `transform(input_grid)` function.

{task_description}

Focus on what stays the same and what changes between input and output.
"""

INIT_VARIANTS = [INIT_DIRECT, INIT_PATTERN, INIT_DECOMPOSE, INIT_ANALOGY, INIT_CONSTRAINT]


MUTATE_BUG_FIX = """The following transform function has bugs. Fix them.

Current code:
```python
{code}
```

Errors encountered:
{errors}

{task_description}

Return the fixed `transform(input_grid)` function.
"""

MUTATE_REFINEMENT = """Refine this transform function to improve accuracy.

Current code:
```python
{code}
```

Current accuracy: {accuracy:.1%}

{task_description}

Return the improved `transform(input_grid)` function.
"""

MUTATE_RESTRUCTURE = """Restructure this transform function for better logic flow.

Current code:
```python
{code}
```

{task_description}

Return the restructured `transform(input_grid)` function.
"""

MUTATE_SIMPLIFY = """Simplify this transform function while maintaining correctness.

Current code:
```python
{code}
```

{task_description}

Return the simplified `transform(input_grid)` function.
"""

MUTATE_GENERALIZE = """Generalize this transform function to handle edge cases better.

Current code:
```python
{code}
```

{task_description}

Return the generalized `transform(input_grid)` function.
"""

MUTATION_TEMPLATES = {
    "BUG_FIX": MUTATE_BUG_FIX,
    "REFINEMENT": MUTATE_REFINEMENT,
    "RESTRUCTURE": MUTATE_RESTRUCTURE,
    "SIMPLIFY": MUTATE_SIMPLIFY,
    "GENERALIZE": MUTATE_GENERALIZE,
}


CROSSOVER_TEMPLATE = """Combine the best parts of these two transform functions into one.

Function A (accuracy: {accuracy_a:.1%}):
```python
{code_a}
```

Function B (accuracy: {accuracy_b:.1%}):
```python
{code_b}
```

{task_description}

{complementarity_analysis}

Return a combined `transform(input_grid)` function taking the best of both.
"""


ERROR_ANALYSIS_TEMPLATE = """Analyze the errors in this transform function:

Code:
```python
{code}
```

Errors:
{errors}

Expected vs Actual outputs:
{comparisons}

Provide a structured analysis of what's wrong and how to fix it.
"""
