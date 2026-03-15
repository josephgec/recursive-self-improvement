"""Built-in math task collection."""

from __future__ import annotations

from src.core.executor import Task


class MathTaskLoader:
    """Loads built-in math tasks."""

    def load(self) -> list[Task]:
        """Return 50+ built-in math tasks with answers."""
        tasks = [
            # Arithmetic
            Task(task_id="math_001", question="What is 15 + 27?", expected_answer="42", domain="math", category="arithmetic"),
            Task(task_id="math_002", question="What is 144 / 12?", expected_answer="12", domain="math", category="arithmetic"),
            Task(task_id="math_003", question="What is 13 * 17?", expected_answer="221", domain="math", category="arithmetic"),
            Task(task_id="math_004", question="What is 1000 - 637?", expected_answer="363", domain="math", category="arithmetic"),
            Task(task_id="math_005", question="What is 2^10?", expected_answer="1024", domain="math", category="arithmetic"),
            Task(task_id="math_006", question="What is 99 * 99?", expected_answer="9801", domain="math", category="arithmetic"),
            Task(task_id="math_007", question="What is 256 / 16?", expected_answer="16", domain="math", category="arithmetic"),
            Task(task_id="math_008", question="What is 47 + 53?", expected_answer="100", domain="math", category="arithmetic"),
            Task(task_id="math_009", question="What is 3^5?", expected_answer="243", domain="math", category="arithmetic"),
            Task(task_id="math_010", question="What is 1234 + 4321?", expected_answer="5555", domain="math", category="arithmetic"),

            # Algebra
            Task(task_id="math_011", question="Solve for x: 2x + 5 = 17", expected_answer="6", domain="math", category="algebra"),
            Task(task_id="math_012", question="Solve for x: 3x - 7 = 14", expected_answer="7", domain="math", category="algebra"),
            Task(task_id="math_013", question="Solve for x: x^2 = 49", expected_answer="7", domain="math", category="algebra"),
            Task(task_id="math_014", question="What is the sum of the roots of x^2 - 5x + 6 = 0?", expected_answer="5", domain="math", category="algebra"),
            Task(task_id="math_015", question="If f(x) = 2x + 3, what is f(5)?", expected_answer="13", domain="math", category="algebra"),
            Task(task_id="math_016", question="Solve for x: 5x = 45", expected_answer="9", domain="math", category="algebra"),
            Task(task_id="math_017", question="What is the product of the roots of x^2 - 5x + 6 = 0?", expected_answer="6", domain="math", category="algebra"),
            Task(task_id="math_018", question="Solve for x: x/4 + 3 = 7", expected_answer="16", domain="math", category="algebra"),
            Task(task_id="math_019", question="If g(x) = x^2 - 1, what is g(4)?", expected_answer="15", domain="math", category="algebra"),
            Task(task_id="math_020", question="Solve: 2(x + 3) = 16", expected_answer="5", domain="math", category="algebra"),

            # Geometry
            Task(task_id="math_021", question="What is the area of a circle with radius 5? Give answer as a multiple of pi.", expected_answer="25", domain="math", category="geometry"),
            Task(task_id="math_022", question="What is the perimeter of a square with side length 8?", expected_answer="32", domain="math", category="geometry"),
            Task(task_id="math_023", question="What is the area of a triangle with base 10 and height 6?", expected_answer="30", domain="math", category="geometry"),
            Task(task_id="math_024", question="How many degrees are in a triangle?", expected_answer="180", domain="math", category="geometry"),
            Task(task_id="math_025", question="What is the hypotenuse of a right triangle with legs 3 and 4?", expected_answer="5", domain="math", category="geometry"),
            Task(task_id="math_026", question="What is the area of a rectangle with length 12 and width 5?", expected_answer="60", domain="math", category="geometry"),
            Task(task_id="math_027", question="What is the circumference of a circle with radius 7? Express as a multiple of pi.", expected_answer="14", domain="math", category="geometry"),
            Task(task_id="math_028", question="How many sides does a hexagon have?", expected_answer="6", domain="math", category="geometry"),
            Task(task_id="math_029", question="What is the volume of a cube with side length 3?", expected_answer="27", domain="math", category="geometry"),
            Task(task_id="math_030", question="What is the sum of interior angles of a pentagon?", expected_answer="540", domain="math", category="geometry"),

            # Number theory
            Task(task_id="math_031", question="What is the GCD of 12 and 18?", expected_answer="6", domain="math", category="number_theory"),
            Task(task_id="math_032", question="What is the LCM of 4 and 6?", expected_answer="12", domain="math", category="number_theory"),
            Task(task_id="math_033", question="Is 17 a prime number? Answer yes or no.", expected_answer="yes", domain="math", category="number_theory"),
            Task(task_id="math_034", question="What is 10 factorial (10!)?", expected_answer="3628800", domain="math", category="number_theory"),
            Task(task_id="math_035", question="What is the remainder when 27 is divided by 4?", expected_answer="3", domain="math", category="number_theory"),
            Task(task_id="math_036", question="What is the smallest prime number?", expected_answer="2", domain="math", category="number_theory"),
            Task(task_id="math_037", question="How many prime numbers are there between 1 and 20?", expected_answer="8", domain="math", category="number_theory"),
            Task(task_id="math_038", question="What is 7! (7 factorial)?", expected_answer="5040", domain="math", category="number_theory"),
            Task(task_id="math_039", question="What is the next prime after 23?", expected_answer="29", domain="math", category="number_theory"),
            Task(task_id="math_040", question="What is the sum of the first 10 positive integers?", expected_answer="55", domain="math", category="number_theory"),

            # Sequences and series
            Task(task_id="math_041", question="What is the 10th term of the arithmetic sequence 2, 5, 8, 11, ...?", expected_answer="29", domain="math", category="sequences"),
            Task(task_id="math_042", question="What is the sum of the first 5 terms of the geometric series 1, 2, 4, 8, 16?", expected_answer="31", domain="math", category="sequences"),
            Task(task_id="math_043", question="What is the next number in the Fibonacci sequence: 1, 1, 2, 3, 5, 8, 13, ?", expected_answer="21", domain="math", category="sequences"),
            Task(task_id="math_044", question="What is the 7th term of the sequence 3, 6, 12, 24, ...?", expected_answer="192", domain="math", category="sequences"),
            Task(task_id="math_045", question="What is the sum of the first 100 positive integers?", expected_answer="5050", domain="math", category="sequences"),

            # Probability and combinatorics
            Task(task_id="math_046", question="How many ways can you arrange 3 books on a shelf?", expected_answer="6", domain="math", category="combinatorics"),
            Task(task_id="math_047", question="What is C(5,2) (5 choose 2)?", expected_answer="10", domain="math", category="combinatorics"),
            Task(task_id="math_048", question="What is C(6,3)?", expected_answer="20", domain="math", category="combinatorics"),
            Task(task_id="math_049", question="How many ways can you flip 3 coins?", expected_answer="8", domain="math", category="combinatorics"),
            Task(task_id="math_050", question="What is P(4,2) (4 permutation 2)?", expected_answer="12", domain="math", category="combinatorics"),

            # Word problems
            Task(task_id="math_051", question="A train travels at 60 mph. How far does it go in 2.5 hours?", expected_answer="150", domain="math", category="word_problems"),
            Task(task_id="math_052", question="If a shirt costs $25 and is 20% off, what is the sale price?", expected_answer="20", domain="math", category="word_problems"),
            Task(task_id="math_053", question="A recipe needs 3 cups of flour for 12 cookies. How many cups for 20 cookies?", expected_answer="5", domain="math", category="word_problems"),
            Task(task_id="math_054", question="If 5 workers can build a wall in 10 days, how many days for 10 workers?", expected_answer="5", domain="math", category="word_problems"),
            Task(task_id="math_055", question="A car uses 5 gallons of gas to travel 150 miles. How many miles per gallon?", expected_answer="30", domain="math", category="word_problems"),
        ]
        return tasks
