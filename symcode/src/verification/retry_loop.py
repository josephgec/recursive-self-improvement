"""Retry loop: generate -> execute -> verify -> retry up to k times."""

from __future__ import annotations

import time
from typing import Any

from src.pipeline.answer_extractor import AnswerExtractor
from src.pipeline.code_generator import SymCodeGenerator
from src.pipeline.output_parser import CodeBlockParser
from src.pipeline.router import TaskType
from src.verification.answer_checker import AnswerChecker
from src.verification.executor import SymCodeExecutor
from src.verification.feedback_generator import FeedbackGenerator
from src.verification.result_types import AttemptRecord, CodeExecutionResult, SolveResult
from src.utils.logging import get_logger

logger = get_logger("retry_loop")


class RetryLoop:
    """Orchestrate the generate -> execute -> verify -> retry loop."""

    def __init__(
        self,
        generator: SymCodeGenerator,
        executor: SymCodeExecutor | None = None,
        checker: AnswerChecker | None = None,
        feedback_gen: FeedbackGenerator | None = None,
        extractor: AnswerExtractor | None = None,
        max_retries: int = 3,
    ):
        self.generator = generator
        self.executor = executor or SymCodeExecutor()
        self.checker = checker or AnswerChecker()
        self.feedback_gen = feedback_gen or FeedbackGenerator()
        self.extractor = extractor or AnswerExtractor()
        self.parser = CodeBlockParser()
        self.max_retries = max_retries

    def solve(
        self,
        problem: str,
        task_type: TaskType | None = None,
        expected_answer: str | None = None,
    ) -> SolveResult:
        """Run the full solve loop with retries.

        1. Generate code
        2. Execute it
        3. Check answer (if expected_answer provided)
        4. If wrong or error, generate feedback and retry

        Returns SolveResult with all attempt records.
        """
        start_time = time.time()
        attempts: list[AttemptRecord] = []
        last_code = ""
        last_feedback = ""

        for attempt_num in range(1, self.max_retries + 1):
            # ── Generate ────────────────────────────────────────────
            if attempt_num == 1:
                gen_result = self.generator.generate(
                    problem, task_type, use_cache=True
                )
            else:
                gen_result = self.generator.generate_correction(
                    problem,
                    last_code,
                    last_feedback,
                    attempt=attempt_num,
                    max_attempts=self.max_retries,
                    use_cache=False,
                )

            code = gen_result.code
            last_code = code

            # ── Validate structure ──────────────────────────────────
            valid, issues = self.parser.validate_structure(code)
            if not valid and not code.strip():
                exec_result = CodeExecutionResult(
                    success=False,
                    error=None,
                )
                record = AttemptRecord(
                    attempt_number=attempt_num,
                    code=code,
                    execution_result=exec_result,
                    feedback="No code was generated.",
                    generation_model=gen_result.model,
                )
                attempts.append(record)
                last_feedback = "No code was generated. Please write a complete Python solution."
                continue

            # ── Execute ─────────────────────────────────────────────
            exec_result = self.executor.execute(code)

            # ── Extract answer ──────────────────────────────────────
            extracted = None
            extracted_str = None

            if exec_result.success:
                extracted = self.extractor.extract_from_execution(
                    namespace={"answer": exec_result.answer}
                    if exec_result.answer is not None
                    else None,
                    stdout=exec_result.stdout,
                )
                if extracted:
                    extracted_str = extracted.normalized

            # ── Check answer ────────────────────────────────────────
            answer_correct = None
            if extracted_str is not None and expected_answer is not None:
                answer_correct = self.checker.check(extracted_str, expected_answer)

            # ── Build attempt record ────────────────────────────────
            feedback = ""
            if not exec_result.success:
                feedback = self.feedback_gen.generate(exec_result, code)
            elif answer_correct is False and expected_answer is not None:
                feedback = self.feedback_gen.generate_wrong_answer_feedback(
                    got=extracted_str or "(none)",
                    expected=expected_answer,
                    code=code,
                )
            elif exec_result.success and extracted_str is None:
                feedback = (
                    "Code executed but no answer was found. "
                    "Make sure to assign the result to a variable named 'answer'."
                )

            record = AttemptRecord(
                attempt_number=attempt_num,
                code=code,
                execution_result=exec_result,
                extracted_answer=extracted_str,
                answer_correct=answer_correct,
                feedback=feedback,
                generation_model=gen_result.model,
            )
            attempts.append(record)
            last_feedback = feedback

            # ── Success? ────────────────────────────────────────────
            if exec_result.success and extracted_str is not None:
                if expected_answer is None or answer_correct:
                    # Success!
                    break

            # If wrong answer but code ran, continue to retry
            # If execution error, continue to retry

        # ── Build final result ──────────────────────────────────────
        elapsed = time.time() - start_time

        final_answer = None
        correct = False
        for rec in reversed(attempts):
            if rec.extracted_answer is not None:
                final_answer = rec.extracted_answer
                correct = bool(rec.answer_correct)
                break

        return SolveResult(
            problem=problem,
            expected_answer=expected_answer,
            final_answer=final_answer,
            correct=correct,
            num_attempts=len(attempts),
            attempts=attempts,
            task_type=task_type.value if task_type else "",
            pipeline="symcode",
            total_time=elapsed,
        )
