"""Built-in code/programming task collection."""

from __future__ import annotations

from src.core.executor import Task


class CodeTaskLoader:
    """Loads built-in code/programming tasks."""

    def load(self) -> list[Task]:
        """Return 20+ built-in code tasks with answers."""
        tasks = [
            Task(
                task_id="code_001",
                question="What is the output of: print(len('hello'))",
                expected_answer="5",
                domain="code",
                category="strings",
            ),
            Task(
                task_id="code_002",
                question="What is the output of: print(list(range(5)))",
                expected_answer="[0, 1, 2, 3, 4]",
                domain="code",
                category="lists",
            ),
            Task(
                task_id="code_003",
                question="What is the output of: print(2 ** 8)",
                expected_answer="256",
                domain="code",
                category="arithmetic",
            ),
            Task(
                task_id="code_004",
                question="What is the output of: print('hello'[::-1])",
                expected_answer="olleh",
                domain="code",
                category="strings",
            ),
            Task(
                task_id="code_005",
                question="What is the output of: print(sorted([3,1,4,1,5]))",
                expected_answer="[1, 1, 3, 4, 5]",
                domain="code",
                category="lists",
            ),
            Task(
                task_id="code_006",
                question="What is the output of: print(bool([]))",
                expected_answer="False",
                domain="code",
                category="types",
            ),
            Task(
                task_id="code_007",
                question="What is the output of: print(type(42).__name__)",
                expected_answer="int",
                domain="code",
                category="types",
            ),
            Task(
                task_id="code_008",
                question="What is the output of: print(sum([1,2,3,4,5]))",
                expected_answer="15",
                domain="code",
                category="lists",
            ),
            Task(
                task_id="code_009",
                question="What is the output of: print('abc' * 3)",
                expected_answer="abcabcabc",
                domain="code",
                category="strings",
            ),
            Task(
                task_id="code_010",
                question="What is the output of: print(max([3,7,2,9,4]))",
                expected_answer="9",
                domain="code",
                category="lists",
            ),
            Task(
                task_id="code_011",
                question="What is the output of: print({1,2,3} & {2,3,4})",
                expected_answer="{2, 3}",
                domain="code",
                category="sets",
            ),
            Task(
                task_id="code_012",
                question="What is the output of: print(10 // 3)",
                expected_answer="3",
                domain="code",
                category="arithmetic",
            ),
            Task(
                task_id="code_013",
                question="What is the output of: print(10 % 3)",
                expected_answer="1",
                domain="code",
                category="arithmetic",
            ),
            Task(
                task_id="code_014",
                question="What is the output of: print('hello world'.split())",
                expected_answer="['hello', 'world']",
                domain="code",
                category="strings",
            ),
            Task(
                task_id="code_015",
                question="What is the output of: print(list(zip([1,2],[3,4])))",
                expected_answer="[(1, 3), (2, 4)]",
                domain="code",
                category="lists",
            ),
            Task(
                task_id="code_016",
                question="What is the output of: print(all([True, True, False]))",
                expected_answer="False",
                domain="code",
                category="logic",
            ),
            Task(
                task_id="code_017",
                question="What is the output of: print(any([False, False, True]))",
                expected_answer="True",
                domain="code",
                category="logic",
            ),
            Task(
                task_id="code_018",
                question="What is the output of: print(abs(-42))",
                expected_answer="42",
                domain="code",
                category="arithmetic",
            ),
            Task(
                task_id="code_019",
                question="What is the output of: print(round(3.14159, 2))",
                expected_answer="3.14",
                domain="code",
                category="arithmetic",
            ),
            Task(
                task_id="code_020",
                question="What is the output of: print(len({1:2, 3:4, 5:6}))",
                expected_answer="3",
                domain="code",
                category="dicts",
            ),
            Task(
                task_id="code_021",
                question="What is the output of: print([x**2 for x in range(4)])",
                expected_answer="[0, 1, 4, 9]",
                domain="code",
                category="comprehensions",
            ),
            Task(
                task_id="code_022",
                question="What is the output of: print('HELLO'.lower())",
                expected_answer="hello",
                domain="code",
                category="strings",
            ),
            Task(
                task_id="code_023",
                question="What is the output of: print(bin(10))",
                expected_answer="0b1010",
                domain="code",
                category="types",
            ),
            Task(
                task_id="code_024",
                question="What is the output of: print(chr(65))",
                expected_answer="A",
                domain="code",
                category="types",
            ),
            Task(
                task_id="code_025",
                question="What is the output of: print(ord('A'))",
                expected_answer="65",
                domain="code",
                category="types",
            ),
        ]
        return tasks
