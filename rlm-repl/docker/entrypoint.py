"""Docker sandbox entrypoint: stdin/stdout JSON protocol."""

import io
import json
import sys
import traceback
from contextlib import redirect_stdout, redirect_stderr


def execute_code(code: str, namespace: dict) -> dict:
    """Execute code and return result as dict."""
    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()
    error = None
    error_type = None

    try:
        with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
            exec(compile(code, "<sandbox>", "exec"), namespace)
    except Exception as e:
        error = str(e)
        error_type = type(e).__name__
        stderr_buf.write(traceback.format_exc())

    return {
        "stdout": stdout_buf.getvalue(),
        "stderr": stderr_buf.getvalue(),
        "error": error,
        "error_type": error_type,
    }


def main():
    """Main loop: read JSON commands from stdin, write results to stdout."""
    namespace = {"__builtins__": __builtins__}

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            command = json.loads(line)
        except json.JSONDecodeError:
            result = {"error": "Invalid JSON", "error_type": "JSONDecodeError"}
            print(json.dumps(result), flush=True)
            continue

        action = command.get("action", "execute")

        if action == "execute":
            code = command.get("code", "")
            result = execute_code(code, namespace)
        elif action == "get_variable":
            name = command.get("name", "")
            if name in namespace:
                result = {"value": repr(namespace[name])}
            else:
                result = {"error": f"Variable '{name}' not found"}
        elif action == "set_variable":
            name = command.get("name", "")
            value = command.get("value")
            namespace[name] = value
            result = {"ok": True}
        elif action == "list_variables":
            names = [
                k for k in namespace
                if not k.startswith("_") and k != "__builtins__"
            ]
            result = {"variables": sorted(names)}
        elif action == "reset":
            namespace.clear()
            namespace["__builtins__"] = __builtins__
            result = {"ok": True}
        elif action == "shutdown":
            result = {"ok": True}
            print(json.dumps(result), flush=True)
            break
        else:
            result = {"error": f"Unknown action: {action}"}

        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
