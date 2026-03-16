"""LoCoDiff benchmark: code-focused long-context tasks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class LoCoDiffTask:
    """A single LoCoDiff benchmark task."""
    task_id: str
    category: str  # "function_find", "bug_detect", "diff_apply", "dependency"
    query: str
    context: str
    expected_answer: str
    difficulty: str = "medium"
    metadata: Dict[str, Any] = field(default_factory=dict)


class LoCoDiffBenchmark:
    """Built-in LoCoDiff-style benchmark with 10+ code-focused tasks."""

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        self._tasks: Optional[List[LoCoDiffTask]] = None

    @property
    def tasks(self) -> List[LoCoDiffTask]:
        if self._tasks is None:
            self._tasks = self._build_tasks()
        return self._tasks

    def get_by_category(self, category: str) -> List[LoCoDiffTask]:
        return [t for t in self.tasks if t.category == category]

    def _build_tasks(self) -> List[LoCoDiffTask]:
        tasks: List[LoCoDiffTask] = []

        # Function-finding tasks
        code_base = self._generate_code_context()

        tasks.append(LoCoDiffTask(
            task_id="locodiff_func_1",
            category="function_find",
            query="Find the function named 'calculate_metrics' and return its line number.",
            context=code_base,
            expected_answer="15",
            difficulty="easy",
        ))
        tasks.append(LoCoDiffTask(
            task_id="locodiff_func_2",
            category="function_find",
            query="How many functions are defined in this code?",
            context=code_base,
            expected_answer="5",
            difficulty="easy",
        ))
        tasks.append(LoCoDiffTask(
            task_id="locodiff_func_3",
            category="function_find",
            query="What does the function 'process_data' return?",
            context=code_base,
            expected_answer="processed",
            difficulty="medium",
        ))

        # Bug detection tasks
        buggy_code = self._generate_buggy_code()
        tasks.append(LoCoDiffTask(
            task_id="locodiff_bug_1",
            category="bug_detect",
            query="Find the off-by-one error in this code. Which line has the bug?",
            context=buggy_code,
            expected_answer="8",
            difficulty="medium",
        ))
        tasks.append(LoCoDiffTask(
            task_id="locodiff_bug_2",
            category="bug_detect",
            query="Find the undefined variable in this code.",
            context=buggy_code,
            expected_answer="result_final",
            difficulty="medium",
        ))

        # Diff application tasks
        diff_context = self._generate_diff_context()
        tasks.append(LoCoDiffTask(
            task_id="locodiff_diff_1",
            category="diff_apply",
            query="How many lines were added in this diff?",
            context=diff_context,
            expected_answer="3",
            difficulty="easy",
        ))
        tasks.append(LoCoDiffTask(
            task_id="locodiff_diff_2",
            category="diff_apply",
            query="How many lines were removed in this diff?",
            context=diff_context,
            expected_answer="2",
            difficulty="easy",
        ))
        tasks.append(LoCoDiffTask(
            task_id="locodiff_diff_3",
            category="diff_apply",
            query="What function was modified in this diff?",
            context=diff_context,
            expected_answer="validate_input",
            difficulty="medium",
        ))

        # Dependency analysis tasks
        dep_context = self._generate_dependency_context()
        tasks.append(LoCoDiffTask(
            task_id="locodiff_dep_1",
            category="dependency",
            query="Which module does 'DataProcessor' depend on?",
            context=dep_context,
            expected_answer="utils",
            difficulty="medium",
        ))
        tasks.append(LoCoDiffTask(
            task_id="locodiff_dep_2",
            category="dependency",
            query="How many import statements are in this code?",
            context=dep_context,
            expected_answer="4",
            difficulty="easy",
        ))
        tasks.append(LoCoDiffTask(
            task_id="locodiff_dep_3",
            category="dependency",
            query="What is the class hierarchy? (parent -> child)",
            context=dep_context,
            expected_answer="BaseProcessor -> DataProcessor",
            difficulty="hard",
        ))

        return tasks

    @staticmethod
    def _generate_code_context() -> str:
        return """\
import os
import sys
from typing import List, Dict

# Utility functions
def load_data(path: str) -> Dict:
    with open(path) as f:
        return json.load(f)

def validate(data: Dict) -> bool:
    return "key" in data

# Core metrics
def calculate_metrics(data: Dict) -> Dict:
    total = sum(data.values())
    avg = total / len(data) if data else 0
    return {"total": total, "average": avg}

def process_data(raw: List) -> str:
    filtered = [x for x in raw if x > 0]
    processed = sum(filtered)
    return "processed"

def main():
    data = load_data("input.json")
    if validate(data):
        metrics = calculate_metrics(data)
        result = process_data(list(data.values()))
        print(metrics, result)
"""

    @staticmethod
    def _generate_buggy_code() -> str:
        return """\
def compute_sum(items):
    total = 0
    for i in range(len(items)):
        total += items[i]
    return total

def find_max(items):
    max_val = items[0]
    for i in range(1, len(items) + 1):  # BUG: off-by-one on line 8
        if items[i] > max_val:
            max_val = items[i]
    return max_val

def format_output(data):
    header = "Results:"
    body = str(data)
    return header + result_final  # BUG: undefined variable 'result_final'
"""

    @staticmethod
    def _generate_diff_context() -> str:
        return """\
--- a/validator.py
+++ b/validator.py
@@ -10,8 +10,9 @@ class Validator:

 def validate_input(self, data):
-    if data is None:
-        return False
+    if data is None or len(data) == 0:
+        raise ValueError("Input must not be empty")
+        return self._detailed_validate(data)
+    self.validated = True
     return True
"""

    @staticmethod
    def _generate_dependency_context() -> str:
        return """\
import os
import json
from typing import Optional
from utils import helper_function

class BaseProcessor:
    def __init__(self):
        self.config = {}

    def setup(self):
        self.config = json.loads(os.environ.get("CONFIG", "{}"))

class DataProcessor(BaseProcessor):
    def __init__(self):
        super().__init__()
        self.data = None

    def process(self, input_data: Optional[dict] = None):
        self.setup()
        result = helper_function(input_data or self.data)
        return result
"""

    def __len__(self) -> int:
        return len(self.tasks)

    def __iter__(self):
        return iter(self.tasks)
