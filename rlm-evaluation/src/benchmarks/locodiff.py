"""LoCoDiff benchmark: code diff understanding tasks."""

from __future__ import annotations

import random
from typing import List

from src.benchmarks.task import EvalTask


class LoCoDiffBenchmark:
    """Benchmark for understanding code diffs in long context."""

    name = "locodiff"

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed

    def load(self) -> List[EvalTask]:
        """Load all LoCoDiff tasks."""
        tasks: List[EvalTask] = []
        tasks.extend(self._function_change_tasks())
        tasks.extend(self._bug_fix_tasks())
        tasks.extend(self._refactoring_tasks())
        return tasks

    def _generate_diff_context(self, diff_blocks: List[str], seed_val: int = 0) -> str:
        """Generate a realistic-looking diff context with surrounding code."""
        rng = random.Random(self.seed + seed_val)
        filler_lines = [
            "    # No changes needed here",
            "    pass",
            "    return result",
            "    x = self._process(data)",
            "    logger.info('Processing complete')",
            "    if condition:",
            "        handle_case()",
            "    for item in items:",
            "        process(item)",
            "    self.state = new_state",
        ]
        blocks = []
        for diff in diff_blocks:
            num_before = rng.randint(10, 30)
            num_after = rng.randint(10, 30)
            before_lines = "\n".join(rng.choices(filler_lines, k=num_before))
            after_lines = "\n".join(rng.choices(filler_lines, k=num_after))
            blocks.append(f"{before_lines}\n{diff}\n{after_lines}")
        return "\n\n".join(blocks)

    def _function_change_tasks(self) -> List[EvalTask]:
        """Tasks about identifying function changes in diffs."""
        tasks = []

        diffs = [
            (
                "--- a/src/utils.py\n+++ b/src/utils.py\n@@ -10,7 +10,7 @@\n"
                "-def calculate_total(items):\n"
                "-    return sum(item.price for item in items)\n"
                "+def calculate_total(items, tax_rate=0.0):\n"
                "+    subtotal = sum(item.price for item in items)\n"
                "+    return subtotal * (1 + tax_rate)",
                "What parameter was added to calculate_total?",
                "tax_rate",
            ),
            (
                "--- a/src/auth.py\n+++ b/src/auth.py\n@@ -25,5 +25,8 @@\n"
                "-def verify_token(token):\n"
                "-    return jwt.decode(token, SECRET)\n"
                "+def verify_token(token, algorithms=None):\n"
                "+    if algorithms is None:\n"
                "+        algorithms = ['HS256']\n"
                "+    return jwt.decode(token, SECRET, algorithms=algorithms)",
                "What default algorithm is used in verify_token?",
                "HS256",
            ),
            (
                "--- a/src/models.py\n+++ b/src/models.py\n@@ -5,4 +5,6 @@\n"
                " class User:\n"
                "-    def __init__(self, name):\n"
                "+    def __init__(self, name, email=None):\n"
                "         self.name = name\n"
                "+        self.email = email",
                "What new attribute was added to the User class?",
                "email",
            ),
            (
                "--- a/src/config.py\n+++ b/src/config.py\n@@ -1,3 +1,3 @@\n"
                "-MAX_RETRIES = 3\n"
                "+MAX_RETRIES = 5\n"
                " TIMEOUT = 30",
                "What was MAX_RETRIES changed to?",
                "5",
            ),
        ]

        for i, (diff, question, answer) in enumerate(diffs):
            context = self._generate_diff_context([diff], seed_val=100 + i)
            tasks.append(EvalTask(
                task_id=f"locodiff_func_{i}",
                benchmark="locodiff",
                query=question,
                context=context,
                expected_answer=answer,
                category="function_change",
                difficulty="medium",
            ))
        return tasks

    def _bug_fix_tasks(self) -> List[EvalTask]:
        """Tasks about identifying bug fixes in diffs."""
        tasks = []

        diffs = [
            (
                "--- a/src/parser.py\n+++ b/src/parser.py\n@@ -15,5 +15,5 @@\n"
                " def parse_int(s):\n"
                "-    return int(s)\n"
                "+    return int(s.strip())\n",
                "What bug was fixed in parse_int?",
                "strip whitespace before parsing",
            ),
            (
                "--- a/src/database.py\n+++ b/src/database.py\n@@ -30,5 +30,7 @@\n"
                " def get_connection():\n"
                "-    conn = db.connect()\n"
                "-    return conn\n"
                "+    conn = db.connect()\n"
                "+    if conn is None:\n"
                "+        raise ConnectionError('Failed to connect')\n"
                "+    return conn",
                "What check was added to get_connection?",
                "null check on connection",
            ),
            (
                "--- a/src/cache.py\n+++ b/src/cache.py\n@@ -8,5 +8,5 @@\n"
                " def get_cached(key):\n"
                "-    if key in cache:\n"
                "+    if key in cache and not cache[key].expired:\n"
                "         return cache[key].value",
                "What condition was added to the cache lookup?",
                "expiration check",
            ),
        ]

        for i, (diff, question, answer) in enumerate(diffs):
            context = self._generate_diff_context([diff], seed_val=200 + i)
            tasks.append(EvalTask(
                task_id=f"locodiff_bugfix_{i}",
                benchmark="locodiff",
                query=question,
                context=context,
                expected_answer=answer,
                category="bug_fix",
                difficulty="medium",
            ))
        return tasks

    def _refactoring_tasks(self) -> List[EvalTask]:
        """Tasks about identifying refactoring patterns in diffs."""
        tasks = []

        diffs = [
            (
                "--- a/src/handler.py\n+++ b/src/handler.py\n@@ -5,10 +5,6 @@\n"
                "-def handle_get(request):\n"
                "-    data = fetch_data(request.params)\n"
                "-    return Response(data)\n"
                "-\n"
                "-def handle_post(request):\n"
                "-    data = fetch_data(request.body)\n"
                "-    return Response(data)\n"
                "+def handle_request(request, method='GET'):\n"
                "+    source = request.params if method == 'GET' else request.body\n"
                "+    data = fetch_data(source)\n"
                "+    return Response(data)",
                "How many functions were merged in the refactoring?",
                "2",
            ),
            (
                "--- a/src/validator.py\n+++ b/src/validator.py\n@@ -1,8 +1,4 @@\n"
                "-def validate_name(name):\n"
                "-    if not name:\n"
                "-        raise ValueError('name required')\n"
                "-    if len(name) > 100:\n"
                "-        raise ValueError('name too long')\n"
                "+def validate_name(name):\n"
                "+    _validate_string(name, 'name', max_len=100)",
                "What was the validation logic extracted into?",
                "_validate_string",
            ),
            (
                "--- a/src/api.py\n+++ b/src/api.py\n@@ -10,6 +10,6 @@\n"
                " class APIClient:\n"
                "-    def __init__(self):\n"
                "-        self.session = requests.Session()\n"
                "+    def __init__(self, session=None):\n"
                "+        self.session = session or requests.Session()",
                "What design pattern was introduced to APIClient?",
                "dependency injection",
            ),
        ]

        for i, (diff, question, answer) in enumerate(diffs):
            context = self._generate_diff_context([diff], seed_val=300 + i)
            tasks.append(EvalTask(
                task_id=f"locodiff_refactor_{i}",
                benchmark="locodiff",
                query=question,
                context=context,
                expected_answer=answer,
                category="refactoring",
                difficulty="hard",
            ))
        return tasks
