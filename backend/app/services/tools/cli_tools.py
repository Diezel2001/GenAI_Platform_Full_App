from __future__ import annotations

import os
import sys
import shlex
import signal
import subprocess
import textwrap

from typing import Optional

from pydantic import BaseModel, Field


# =========================================================
# SCHEMAS
# =========================================================

class ExecuteCommandSchema(BaseModel):
    command: str = Field(
        ...,
        description=(
            "Shell command to execute. "
            "Can include pipes (|), redirects (>), chaining (&&), etc. "
            "The command is run through a shell by default."
        ),
    )
    cwd: Optional[str] = Field(
        None,
        description=(
            "Working directory for the command. "
            "Defaults to the project root if not provided."
        ),
    )
    timeout_seconds: int = Field(
        60, ge=1, le=600,
        description="Max execution time in seconds (1–600).",
    )
    shell: bool = Field(
        True,
        description=(
            "If True (default), run via /bin/sh -c so pipes, "
            "redirects, chaining etc. work. "
            "If False, the command is parsed into argv and executed directly "
            "(safer for simple commands with no shell metacharacters)."
        ),
    )
    env: Optional[dict[str, str]] = Field(
        None,
        description=(
            "Optional dictionary of additional environment variables "
            "to set for the command. Merged on top of the current env."
        ),
    )
    max_output_chars: int = Field(
        10_000, ge=100, le=100_000,
        description="Truncate combined output to this many characters.",
    )


# =========================================================
# IMPLEMENTATION
# =========================================================

def execute_command(
    command: str,
    cwd: Optional[str] = None,
    timeout_seconds: int = 60,
    shell: bool = True,
    env: Optional[dict[str, str]] = None,
    max_output_chars: int = 10_000,
) -> str:
    """
    Execute a CLI command and return its combined stdout/stderr output.

    Uses subprocess with configurable timeout, optional shell mode,
    working directory, and environment variables.

    Returns a formatted string with the exit code and captured output
    (or an error message if the command could not be started).
    """

    # Resolve working directory
    if cwd:
        resolved_cwd = os.path.abspath(os.path.expanduser(cwd))
        if not os.path.isdir(resolved_cwd):
            return (
                f"ERROR: Working directory not found or is not a directory:\n"
                f"  {resolved_cwd}"
            )
    else:
        resolved_cwd = os.getcwd()

    # Prepare environment
    full_env = os.environ.copy()
    if env:
        full_env.update(env)

    # Build args
    if shell:
        args = command
    else:
        try:
            args = shlex.split(command)
        except ValueError as e:
            return (
                f"ERROR: Could not parse command into arguments "
                f"(shell=False): {e}"
            )

    try:
        proc = subprocess.Popen(
            args,
            shell=shell,
            cwd=resolved_cwd,
            env=full_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            preexec_fn=_preexec_setpgid if sys.platform != "win32" else None,
        )
    except FileNotFoundError:
        return f"ERROR: Command not found: {command}"
    except PermissionError:
        return f"ERROR: Permission denied when trying to execute: {command}"
    except Exception as e:
        return f"ERROR: Failed to start command: {e}"

    # Collect output with timeout
    try:
        stdout_data, stderr_data = proc.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        _kill_process_group(proc)
        stdout_data, stderr_data = proc.communicate(timeout=5)
        # Build partial output message
        sections = [
            f"TIMEOUT: Command exceeded {timeout_seconds} seconds and was killed.",
            "",
            f"--- partial stdout ---",
            stdout_data.rstrip() if stdout_data else "(none)",
        ]
        if stderr_data and stderr_data.strip():
            sections.append("")
            sections.append(f"--- partial stderr ---")
            sections.append(stderr_data.rstrip())
        result = "\n".join(sections)
        return _truncate(result, max_output_chars)

    exit_code = proc.returncode
    sections = []

    # Standard output
    if stdout_data and stdout_data.strip():
        sections.append(f"--- stdout ---")
        sections.append(stdout_data.rstrip())

    # Standard error
    if stderr_data and stderr_data.strip():
        sections.append(f"--- stderr ---")
        sections.append(stderr_data.rstrip())

    # Exit code summary
    if exit_code == 0:
        sections.append(f"\nExit code: 0 (success)")
    else:
        sections.append(f"\nExit code: {exit_code}")

    result = "\n".join(sections)
    return _truncate(result, max_output_chars)


# =========================================================
# HELPERS
# =========================================================

def _preexec_setpgid() -> None:
    """
    Pre-exec hook that places the child process in its own process group.
    This allows us to kill the entire process tree if the command times out.
    """
    os.setpgrp()


def _kill_process_group(proc: subprocess.Popen) -> None:
    """
    Kill the entire process group of the given process.
    Sends SIGTERM first, then SIGKILL after a short grace period.
    """
    if sys.platform == "win32":
        proc.kill()
        return

    pgid = proc.pid  # Because we called setpgrp, pid == pgid
    try:
        # Try graceful termination first
        os.killpg(pgid, signal.SIGTERM)
        import time
        time.sleep(1)
        # Force kill if still running
        if proc.poll() is None:
            os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        pass  # Process already exited


def _truncate(text: str, max_chars: int) -> str:
    """Truncate text to max_chars, appending a truncation notice."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "\n\n[OUTPUT TRUNCATED — remaining output not shown]"


# =========================================================
# TOOL REGISTRY ENTRIES
# =========================================================

CLI_TOOLS = {
    "execute_command": {
        "func": execute_command,
        "schema": ExecuteCommandSchema,
        "description": (
            "Execute a CLI command and return stdout, stderr, and exit code. "
            "Supports pipes, redirects, chaining via shell=True (default). "
            "Configurable working directory, timeout, and environment variables. "
            "Output is truncated to prevent runaway responses. "
            "Timeout kills the entire process group cleanly."
        ),
    },
}