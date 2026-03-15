"""Z3 SMT solver runner with subprocess isolation."""

from __future__ import annotations

from symbolic.src.executor import run_in_subprocess
from symbolic.src.result_types import Z3Result

# Namespace setup code executed before user code in the subprocess.
_Z3_SETUP = """\
from z3 import *
"""


class Z3Runner:
    """Execute Z3 code and verify logical properties.

    All code runs in a fresh subprocess so that timeouts, segfaults, and
    memory leaks are isolated from the caller.

    Parameters
    ----------
    timeout:
        Maximum wall-clock seconds for a single execution.
    max_memory_mb:
        Soft memory hint (not enforced at OS level in subprocess mode).
    """

    def __init__(self, timeout: float = 30.0, max_memory_mb: int = 2048) -> None:
        self.timeout = timeout
        self.max_memory_mb = max_memory_mb

    def execute(self, code: str) -> Z3Result:
        """Execute Z3 code and return a structured result.

        The code is run with ``from z3 import *`` already available.
        The runner looks for a ``Solver`` instance named ``s`` (or ``solver``)
        and extracts the check result and model.
        """
        wrapped = _Z3_RESULT_WRAPPER.format(user_code=code)

        raw = run_in_subprocess(
            code=wrapped,
            namespace_setup=_Z3_SETUP,
            timeout=self.timeout,
        )

        if not raw["success"]:
            return Z3Result(
                satisfiable=None,
                error=raw["error"],
                execution_time_ms=raw.get("execution_time_ms", 0.0),
            )

        variables = raw["variables"]
        sat_value = variables.get("_sat_result")
        if sat_value == "sat":
            satisfiable = True
        elif sat_value == "unsat":
            satisfiable = False
        else:
            satisfiable = None

        return Z3Result(
            satisfiable=satisfiable,
            model=variables.get("_model_dict", {}),
            proof=variables.get("_proof_str"),
            execution_time_ms=raw.get("execution_time_ms", 0.0),
        )

    def check_implication(
        self,
        premises: list[str],
        conclusion: str,
    ) -> bool:
        """Check whether *premises* logically imply *conclusion*.

        Works by asserting all premises, negating the conclusion, and
        checking for UNSAT.  Returns True if the implication holds.
        """
        code = _IMPLICATION_CHECK.format(
            premises=repr(premises),
            conclusion=repr(conclusion),
        )
        raw = run_in_subprocess(
            code=code,
            namespace_setup=_Z3_SETUP,
            timeout=self.timeout,
        )
        if not raw["success"]:
            return False
        return bool(raw["variables"].get("_implies", False))

    def verify_program_property(
        self,
        preconditions: list[str],
        postconditions: list[str],
        program_constraints: list[str],
    ) -> Z3Result:
        """Verify that program constraints + preconditions entail postconditions.

        Asserts preconditions and program constraints, negates the conjunction
        of postconditions, and checks for UNSAT (i.e., the postconditions
        must hold).
        """
        code = _PROGRAM_PROPERTY_CHECK.format(
            preconditions=repr(preconditions),
            postconditions=repr(postconditions),
            program_constraints=repr(program_constraints),
        )
        raw = run_in_subprocess(
            code=code,
            namespace_setup=_Z3_SETUP,
            timeout=self.timeout,
        )

        if not raw["success"]:
            return Z3Result(
                satisfiable=None,
                error=raw["error"],
                execution_time_ms=raw.get("execution_time_ms", 0.0),
            )

        variables = raw["variables"]
        sat_value = variables.get("_sat_result")
        if sat_value == "unsat":
            # UNSAT means the postconditions are always satisfied
            satisfiable = False
        elif sat_value == "sat":
            satisfiable = True
        else:
            satisfiable = None

        return Z3Result(
            satisfiable=satisfiable,
            model=variables.get("_model_dict", {}),
            proof=variables.get("_proof_str"),
            execution_time_ms=raw.get("execution_time_ms", 0.0),
        )


# ---------------------------------------------------------------------------
# Code templates
# ---------------------------------------------------------------------------

_Z3_RESULT_WRAPPER = """\
# ---- user code ----
{user_code}
# ---- capture result ----
_solver = None
for _name in ('s', 'solver', 'S'):
    if _name in dir() and hasattr(eval(_name), 'check'):
        _solver = eval(_name)
        break

_sat_result = None
_model_dict = {{}}
_proof_str = None

if _solver is not None:
    _check = _solver.check()
    _sat_result = str(_check)
    if _check == sat:
        _m = _solver.model()
        _model_dict = {{str(d): str(_m[d]) for d in _m.decls()}}
    elif _check == unsat:
        try:
            _proof_str = str(_solver.proof())
        except Exception:
            _proof_str = None
"""

_IMPLICATION_CHECK = """\
# Parse premises and conclusion using Z3
_premises_strs = {premises}
_conclusion_str = {conclusion}

# Declare a common integer variable x (extend as needed)
x, y, z, a, b, c, n, m = Ints('x y z a b c n m')

s = Solver()
for _p in _premises_strs:
    s.add(eval(_p))

# Negate the conclusion
s.add(Not(eval(_conclusion_str)))

_check = s.check()
_implies = (_check == unsat)
"""

_PROGRAM_PROPERTY_CHECK = """\
_pre_strs = {preconditions}
_post_strs = {postconditions}
_prog_strs = {program_constraints}

x, y, z, a, b, c, n, m = Ints('x y z a b c n m')

s = Solver()
for _p in _pre_strs:
    s.add(eval(_p))
for _p in _prog_strs:
    s.add(eval(_p))

# Negate the conjunction of postconditions
_post_exprs = [eval(_p) for _p in _post_strs]
s.add(Not(And(*_post_exprs)))

_check = s.check()
_sat_result = str(_check)
_model_dict = {{}}
_proof_str = None

if _check == sat:
    _m = s.model()
    _model_dict = {{str(d): str(_m[d]) for d in _m.decls()}}
elif _check == unsat:
    try:
        _proof_str = str(s.proof())
    except Exception:
        _proof_str = None
"""
