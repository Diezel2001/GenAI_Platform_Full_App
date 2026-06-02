"""
tools/__init__.py

Central registry that assembles all tool modules into a single
dict compatible with WorkerAgent's ToolRuntime.

Usage:
    from app.tools import ALL_TOOLS, get_tools

    # Pass everything to WorkerAgent
    agent = WorkerAgent(llm=llm, tools=ALL_TOOLS, checkpointer=checkpointer)

    # Pass a filtered subset (e.g. only file + web tools)
    agent = WorkerAgent(llm=llm, tools=get_tools("read_file", "web_search"), ...)
"""

from __future__ import annotations

import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


# =========================================================
# IMPORT TOOL MODULES
# Each module exposes a dict: { "tool_name": { func, schema, description } }
# Import errors are caught individually so a missing optional dependency
# (e.g. psycopg2) doesn't break unrelated tools.
# =========================================================

def _safe_import(module_path: str, registry_name: str) -> Dict[str, Any]:
    try:
        import importlib
        mod = importlib.import_module(module_path)
        registry = getattr(mod, registry_name)
        logger.debug("Loaded tool module: %s (%d tools)", module_path, len(registry))
        return registry
    except ImportError as e:
        logger.warning(
            "Could not import tool module '%s': %s — "
            "tools from this module will be unavailable.",
            module_path,
            e,
        )
        return {}
    except Exception as e:
        logger.error(
            "Unexpected error loading tool module '%s': %s",
            module_path,
            e,
        )
        return {}


# Adjust these import paths to match your project structure.
# If your tools folder is at app/tools/, use "app.tools.file_tools" etc.

_FILE_TOOLS      = _safe_import("app.services.tools.file_tools",     "FILE_TOOLS")
_WEB_TOOLS       = _safe_import("app.services.tools.web_tools",      "WEB_TOOLS")
_PYTHON_TOOLS    = _safe_import("app.services.tools.python_tools",   "PYTHON_TOOLS")
_HTTP_TOOLS      = _safe_import("app.services.tools.http_tools",     "HTTP_TOOLS")
_CLI_TOOLS       = _safe_import("app.services.tools.cli_tools",      "CLI_TOOLS")


# =========================================================
# COMBINED REGISTRY
# =========================================================

ALL_TOOLS: Dict[str, Any] = {
    **_FILE_TOOLS,
    **_WEB_TOOLS,
    **_PYTHON_TOOLS,
    **_HTTP_TOOLS,
    **_CLI_TOOLS,
}

logger.info(
    "Tool registry loaded: %d tools available — %s",
    len(ALL_TOOLS),
    sorted(ALL_TOOLS.keys()),
)


# =========================================================
# FILTERED ACCESS
# =========================================================

def get_tools(*tool_names: str) -> Dict[str, Any]:
    """
    Return a subset of ALL_TOOLS by name.
    Logs a warning for any requested tool not found in the registry.

    Example:
        tools = get_tools("read_file", "write_file", "web_search")
    """
    result = {}
    for name in tool_names:
        if name in ALL_TOOLS:
            result[name] = ALL_TOOLS[name]
        else:
            logger.warning(
                "Requested tool '%s' not found in registry. "
                "Available tools: %s",
                name,
                sorted(ALL_TOOLS.keys()),
            )
    return result


def get_tools_for_skill(allowed_tools: list[str]) -> Dict[str, Any]:
    """
    Convenience wrapper — same as get_tools() but accepts the
    allowed_tools list directly from a Skill object.

    Example:
        tools = get_tools_for_skill(skill.metadata.allowed_tools)
    """
    return get_tools(*allowed_tools)


def list_available_tools() -> list[str]:
    """Return a sorted list of all registered tool names."""
    return sorted(ALL_TOOLS.keys())


def describe_tools(tool_names: Optional[list[str]] = None) -> str:
    """
    Return a human-readable description of all tools (or a subset).
    Useful for debugging and for building planner prompts.
    """
    tools = (
        get_tools(*tool_names) if tool_names else ALL_TOOLS
    )

    if not tools:
        return "No tools available."

    lines = []
    for name, tool in sorted(tools.items()):
        desc = tool.get("description", "No description.")
        schema = tool.get("schema")
        fields = []
        if schema:
            for field_name, field_info in schema.model_fields.items():
                required = field_info.is_required()
                annotation = field_info.annotation
                marker = "" if required else "?"
                fields.append(f"{field_name}{marker}: {annotation}")
        fields_str = ", ".join(fields) or "none"
        lines.append(
            f"{'─' * 60}\n"
            f"Tool:   {name}\n"
            f"Desc:   {desc}\n"
            f"Args:   {fields_str}"
        )

    return "\n".join(lines)
