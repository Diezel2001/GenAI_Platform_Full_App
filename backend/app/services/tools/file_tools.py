from __future__ import annotations

import difflib
import os
import json
import shutil
import stat
import chardet
import re

from datetime import datetime
from pathlib import Path
from typing import Optional, List

from pydantic import BaseModel, Field


# =========================================================
# SCHEMAS
# =========================================================

class ReadFileSchema(BaseModel):
    path: str = Field(..., description="Absolute or relative path to the file to read.")
    encoding: Optional[str] = Field(None, description="File encoding. Auto-detected if not provided.")
    max_chars: Optional[int] = Field(None, description="Truncate output to this many characters. None = full file.")


class CreateFileSchema(BaseModel):
    path: str = Field(..., description="Absolute or relative path where the new file will be created.")
    content: str = Field("", description="Text content to write to the new file.")
    encoding: str = Field("utf-8", description="File encoding to use when writing.")
    overwrite: bool = Field(False, description="If False (default), fails when file already exists. Set True to allow overwriting an existing file.")


class WriteFileSchema(BaseModel):
    path: str = Field(..., description="Absolute or relative path to write to. Parent directories are created if missing.")
    content: str = Field(..., description="Text content to write to the file.")
    mode: str = Field("w", description="Write mode: 'w' (overwrite) or 'a' (append).")
    encoding: str = Field("utf-8", description="File encoding to use when writing.")


class CopyFileSchema(BaseModel):
    src: str = Field(..., description="Absolute or relative path to the source file or directory.")
    dst: str = Field(..., description="Absolute or relative path to the destination.")
    overwrite: bool = Field(False, description="If False (default), fails when destination already exists.")


class MoveFileSchema(BaseModel):
    src: str = Field(..., description="Absolute or relative path to the source file or directory.")
    dst: str = Field(..., description="Absolute or relative path to the destination.")
    overwrite: bool = Field(False, description="If False (default), fails when destination already exists.")


class DeleteFileSchema(BaseModel):
    path: str = Field(..., description="Absolute or relative path to the file to delete.")
    confirm: bool = Field(False, description="Must be set to True to confirm deletion. This is a safety guard.")


class CreateDirectorySchema(BaseModel):
    path: str = Field(..., description="Absolute or relative path of the directory to create.")
    exist_ok: bool = Field(True, description="If True (default), does nothing if directory already exists. If False, raises an error.")


class DeleteDirectorySchema(BaseModel):
    path: str = Field(..., description="Absolute or relative path of the directory to delete.")
    recursive: bool = Field(False, description="If True, deletes the directory and all its contents. If False, only deletes empty directories.")
    confirm: bool = Field(False, description="Must be set to True to confirm deletion. This is a safety guard.")


class ListDirectorySchema(BaseModel):
    path: str = Field(..., description="Absolute or relative path to the directory to list.")
    recursive: bool = Field(False, description="If True, list all files recursively.")
    extension_filter: Optional[str] = Field(None, description="Only return files with this extension (e.g. '.py', '.csv').")


class FileInfoSchema(BaseModel):
    path: str = Field(..., description="Absolute or relative path to the file or directory to inspect.")


class SearchInFilesSchema(BaseModel):
    pattern: str = Field(..., description="Text or regex pattern to search for.")
    path: str = Field(".", description="Directory path to search in (defaults to current dir).")
    file_pattern: Optional[str] = Field(None, description="Glob pattern to filter files (e.g. '*.py', '*.md'). If not set, searches all files.")
    regex: bool = Field(False, description="If True, treat pattern as a regex. If False, plain text search.")
    max_matches: int = Field(100, description="Maximum number of matching lines to return.")
    case_sensitive: bool = Field(True, description="If True, case-sensitive search. If False, case-insensitive.")


class FindFilesSchema(BaseModel):
    pattern: str = Field(..., description="Glob pattern to match files (e.g. '**/*.py', '*.txt', 'data/**/*.csv').")
    path: str = Field(".", description="Root directory to search in (defaults to current dir).")
    max_results: int = Field(200, description="Maximum number of results to return.")


class ReadFileLinesSchema(BaseModel):
    path: str = Field(..., description="Absolute or relative path to the file to read.")
    start_line: int = Field(1, ge=1, description="Starting line number (1-based, inclusive).")
    end_line: Optional[int] = Field(None, description="Ending line number (1-based, inclusive). If not set, reads to end of file.")
    encoding: Optional[str] = Field(None, description="File encoding. Auto-detected if not provided.")


class FileDiffSchema(BaseModel):
    path_a: str = Field(..., description="Absolute or relative path to the first file.")
    path_b: str = Field(..., description="Absolute or relative path to the second file.")
    context_lines: int = Field(3, ge=0, description="Number of context lines around each difference.")
    encoding: Optional[str] = Field(None, description="File encoding. Auto-detected if not provided.")


# =========================================================
# HELPERS
# =========================================================

def _resolve(path: str) -> Path:
    return Path(path).expanduser().resolve()


def _detect_encoding(path: Path) -> str:
    raw_bytes = path.read_bytes()
    detected = chardet.detect(raw_bytes)
    return detected.get("encoding") or "utf-8"


def _read_text(path: Path, encoding: Optional[str] = None) -> str:
    if encoding is None:
        encoding = _detect_encoding(path)
    return path.read_text(encoding=encoding, errors="replace")


def _safe_delete_path(path: Path) -> None:
    """Delete a file or directory, handling read-only permissions."""
    def _onerror(func, fpath, exc_info):
        # Make the file writable and retry
        os.chmod(fpath, stat.S_IWRITE)
        func(fpath)
    if path.is_dir():
        shutil.rmtree(path, onerror=_onerror)
    else:
        try:
            path.unlink()
        except PermissionError:
            os.chmod(path, stat.S_IWRITE)
            path.unlink()


def _format_file_size(size_bytes: int) -> str:
    """Return human-readable file size."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024**2:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024**3:
        return f"{size_bytes / 1024**2:.1f} MB"
    else:
        return f"{size_bytes / 1024**3:.2f} GB"


# =========================================================
# IMPLEMENTATIONS
# =========================================================

def read_file(
    path: str,
    encoding: Optional[str] = None,
    max_chars: Optional[int] = None,
) -> str:
    """
    Read a file and return its text content.
    Handles encoding detection automatically.
    Returns an error string on failure so the agent can observe and recover.
    """

    resolved = _resolve(path)

    if not resolved.exists():
        parent = resolved.parent
        available = (
            [f.name for f in parent.iterdir()]
            if parent.exists()
            else []
        )
        return (
            f"ERROR: File not found: {resolved}\n"
            f"Available in {parent}: {available}"
        )

    if not resolved.is_file():
        return f"ERROR: Path is not a file: {resolved}"

    try:
        content = _read_text(resolved, encoding)
    except Exception as e:
        return f"ERROR reading file: {e}"

    size = len(content)

    if max_chars and size > max_chars:
        content = content[:max_chars]
        return (
            f"[File: {resolved} | {size} chars total | "
            f"showing first {max_chars}]\n\n{content}\n\n"
            f"[TRUNCATED — {size - max_chars} chars not shown]"
        )

    return f"[File: {resolved} | {size} chars]\n\n{content}"


def create_file(
    path: str,
    content: str = "",
    encoding: str = "utf-8",
    overwrite: bool = False,
) -> str:
    """
    Create a new file at the specified path.
    Unlike write_file, this is explicitly for *creating* new files.
    By default it will NOT overwrite an existing file unless overwrite=True.
    Parent directories are created automatically.
    """
    resolved = _resolve(path)

    if resolved.exists() and not overwrite:
        return (
            f"ERROR: File already exists: {resolved}\n"
            f"Set overwrite=True if you want to overwrite it. "
            f"Alternatively, use write_file() for appending/overwriting."
        )

    try:
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(content, encoding=encoding)
    except Exception as e:
        return f"ERROR creating file: {e}"

    return (
        f"Created: {resolved}\n"
        f"Size: {len(content.encode(encoding))} bytes\n"
        f"Encoding: {encoding}"
    )


def write_file(
    path: str,
    content: str,
    mode: str = "w",
    encoding: str = "utf-8",
) -> str:
    """
    Write text content to a file.
    Creates parent directories automatically.
    Returns a confirmation or error string.
    """

    if mode not in ("w", "a"):
        return f"ERROR: Invalid mode '{mode}'. Use 'w' or 'a'."

    resolved = _resolve(path)

    try:
        resolved.parent.mkdir(parents=True, exist_ok=True)
        with resolved.open(mode=mode, encoding=encoding) as f:
            f.write(content)
    except Exception as e:
        return f"ERROR writing file: {e}"

    action = "Written" if mode == "w" else "Appended"
    return (
        f"{action}: {resolved}\n"
        f"Size: {len(content.encode(encoding))} bytes\n"
        f"Encoding: {encoding}"
    )


def copy_file(
    src: str,
    dst: str,
    overwrite: bool = False,
) -> str:
    """
    Copy a file or directory from src to dst.
    Supports both file-to-file and directory-to-directory copying.
    """
    src_resolved = _resolve(src)
    dst_resolved = _resolve(dst)

    if not src_resolved.exists():
        return f"ERROR: Source not found: {src_resolved}"

    if dst_resolved.exists() and not overwrite:
        return (
            f"ERROR: Destination already exists: {dst_resolved}\n"
            f"Set overwrite=True to overwrite."
        )

    try:
        dst_resolved.parent.mkdir(parents=True, exist_ok=True)

        if src_resolved.is_dir():
            shutil.copytree(
                src_resolved,
                dst_resolved,
                dirs_exist_ok=overwrite,
            )
            item_type = "Directory"
        else:
            shutil.copy2(src_resolved, dst_resolved)
            item_type = "File"

    except Exception as e:
        return f"ERROR copying {src_resolved} to {dst_resolved}: {e}"

    return f"Copied {item_type}: {src_resolved} → {dst_resolved}"


def move_file(
    src: str,
    dst: str,
    overwrite: bool = False,
) -> str:
    """
    Move or rename a file or directory from src to dst.
    """
    src_resolved = _resolve(src)
    dst_resolved = _resolve(dst)

    if not src_resolved.exists():
        return f"ERROR: Source not found: {src_resolved}"

    if dst_resolved.exists() and not overwrite:
        return (
            f"ERROR: Destination already exists: {dst_resolved}\n"
            f"Set overwrite=True to overwrite."
        )

    try:
        dst_resolved.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src_resolved), str(dst_resolved))
    except Exception as e:
        return f"ERROR moving {src_resolved} to {dst_resolved}: {e}"

    return f"Moved: {src_resolved} → {dst_resolved}"


def delete_file(
    path: str,
    confirm: bool = False,
) -> str:
    """
    Delete a file at the specified path.
    Requires confirm=True as a safety guard against accidental deletion.
    """
    if not confirm:
        return (
            f"ERROR: Deletion not confirmed.\n"
            f"Set confirm=True to proceed with deleting: {path}"
        )

    resolved = _resolve(path)

    if not resolved.exists():
        return f"ERROR: File not found: {resolved}"

    if not resolved.is_file():
        return f"ERROR: Path is not a file: {resolved}"

    try:
        resolved.unlink()
    except Exception as e:
        return f"ERROR deleting file: {e}"

    return f"Deleted file: {resolved}"


def create_directory(
    path: str,
    exist_ok: bool = True,
) -> str:
    """
    Create a directory (and all parent directories).
    Works like mkdir -p.
    """
    resolved = _resolve(path)

    if resolved.exists() and not exist_ok:
        return f"ERROR: Directory already exists: {resolved}. Set exist_ok=True to allow."

    try:
        resolved.mkdir(parents=True, exist_ok=exist_ok)
    except Exception as e:
        return f"ERROR creating directory: {e}"

    if resolved.exists():
        return f"Directory ready: {resolved}"
    else:
        return f"Created directory: {resolved}"


def delete_directory(
    path: str,
    recursive: bool = False,
    confirm: bool = False,
) -> str:
    """
    Delete a directory.
    If recursive=False, only deletes empty directories (like rmdir).
    If recursive=True, deletes the directory and all contents (like rm -rf).
    Requires confirm=True as a safety guard.
    """
    if not confirm:
        return (
            f"ERROR: Deletion not confirmed.\n"
            f"Set confirm=True to proceed with deleting directory: {path}"
        )

    resolved = _resolve(path)

    if not resolved.exists():
        return f"ERROR: Directory not found: {resolved}"

    if not resolved.is_dir():
        return f"ERROR: Path is not a directory: {resolved}"

    try:
        if recursive:
            _safe_delete_path(resolved)
        else:
            resolved.rmdir()  # Only works if empty
    except OSError as e:
        if "Directory not empty" in str(e):
            return (
                f"ERROR: Directory is not empty: {resolved}\n"
                f"Set recursive=True to delete non-empty directories."
            )
        return f"ERROR deleting directory: {e}"
    except Exception as e:
        return f"ERROR deleting directory: {e}"

    return f"Deleted directory: {resolved}"


def list_directory(
    path: str,
    recursive: bool = False,
    extension_filter: Optional[str] = None,
) -> str:
    """
    List files and directories at the given path.
    Returns a formatted tree or error string.
    """

    resolved = _resolve(path)

    if not resolved.exists():
        return f"ERROR: Directory not found: {resolved}"

    if not resolved.is_dir():
        return f"ERROR: Path is not a directory: {resolved}"

    ext = (
        extension_filter.lower()
        if extension_filter and not extension_filter.startswith(".")
        else extension_filter
    )

    entries = []

    if recursive:
        for item in sorted(resolved.rglob("*")):
            if ext and item.suffix.lower() != ext:
                continue
            rel = item.relative_to(resolved)
            kind = "DIR " if item.is_dir() else "FILE"
            size = (
                f"{item.stat().st_size:>10} bytes"
                if item.is_file()
                else ""
            )
            entries.append(f"  {kind}  {rel}  {size}")
    else:
        for item in sorted(resolved.iterdir()):
            if ext and item.is_file() and item.suffix.lower() != ext:
                continue
            kind = "DIR " if item.is_dir() else "FILE"
            size = (
                f"{item.stat().st_size:>10} bytes"
                if item.is_file()
                else ""
            )
            entries.append(f"  {kind}  {item.name}  {size}")

    if not entries:
        return f"Directory is empty (or no files match filter): {resolved}"

    header = (
        f"Directory: {resolved}\n"
        f"Filter: {extension_filter or 'none'} | "
        f"Recursive: {recursive} | "
        f"Entries: {len(entries)}\n"
        + "-" * 60
    )

    return header + "\n" + "\n".join(entries)


def file_info(
    path: str,
) -> str:
    """
    Return metadata information about a file or directory.
    Includes size, modification time, permissions, file type, etc.
    """
    resolved = _resolve(path)

    if not resolved.exists():
        return f"ERROR: Path not found: {resolved}"

    try:
        st = resolved.stat()
        is_dir = resolved.is_dir()
        is_file = resolved.is_file()
        is_symlink = resolved.is_symlink()

        lines = []
        lines.append(f"Path:     {resolved}")
        lines.append(f"Type:     {'Directory' if is_dir else 'File'}")
        if is_symlink:
            lines.append(f"Symlink:  → {resolved.readlink()}")
        lines.append(f"Size:     {_format_file_size(st.st_size)} ({st.st_size} bytes)")
        lines.append(f"Created:  {datetime.fromtimestamp(st.st_ctime).isoformat()}")
        lines.append(f"Modified: {datetime.fromtimestamp(st.st_mtime).isoformat()}")
        lines.append(f"Accessed: {datetime.fromtimestamp(st.st_atime).isoformat()}")
        lines.append(f"Mode:     {stat.filemode(st.st_mode)} ({oct(st.st_mode)})")
        lines.append(f"Owner:    {st.st_uid} (uid) / {st.st_gid} (gid)")
        lines.append(f"Inode:    {st.st_ino}")
        lines.append(f"Hardlinks: {st.st_nlink}")

        if is_dir:
            # Count contents
            items = list(resolved.iterdir())
            files = sum(1 for i in items if i.is_file())
            dirs = sum(1 for i in items if i.is_dir())
            lines.append(f"Contents: {len(items)} items ({files} files, {dirs} dirs)")

        return "\n".join(lines)

    except Exception as e:
        return f"ERROR getting file info: {e}"


def search_in_files(
    pattern: str,
    path: str = ".",
    file_pattern: Optional[str] = None,
    regex: bool = False,
    max_matches: int = 100,
    case_sensitive: bool = True,
) -> str:
    """
    Search for a text or regex pattern across files in a directory.
    Returns matching lines with file paths and line numbers.
    """
    root = _resolve(path)

    if not root.exists():
        return f"ERROR: Path not found: {root}"

    if not root.is_dir():
        return f"ERROR: Path is not a directory: {root}"

    # Compile regex if needed
    if regex:
        flags = 0 if case_sensitive else re.IGNORECASE
        try:
            compiled = re.compile(pattern, flags)
        except re.error as e:
            return f"ERROR: Invalid regex pattern: {e}"
    else:
        compiled = None

    try:
        matches = []
        if file_pattern:
            iterable = root.rglob(file_pattern)
        else:
            iterable = root.rglob("*")

        for file_path in iterable:
            if not file_path.is_file():
                continue

            # Skip binary files
            try:
                is_binary = False
                with file_path.open("rb") as f:
                    chunk = f.read(1024)
                    if b"\x00" in chunk:
                        is_binary = True
                if is_binary:
                    continue
            except Exception:
                continue

            try:
                encoding = _detect_encoding(file_path)
                with file_path.open("r", encoding=encoding, errors="replace") as f:
                    for line_no, line in enumerate(f, 1):
                        line_stripped = line.rstrip("\n\r")

                        if compiled:
                            if compiled.search(line_stripped):
                                pass  # match
                            else:
                                continue
                        else:
                            if case_sensitive:
                                if pattern not in line_stripped:
                                    continue
                            else:
                                if pattern.lower() not in line_stripped.lower():
                                    continue

                        rel_path = file_path.relative_to(root)
                        matches.append(f"{rel_path}:{line_no}: {line_stripped}")

                        if len(matches) >= max_matches:
                            break
            except Exception:
                continue

            if len(matches) >= max_matches:
                break

        if not matches:
            return f"No matches found for '{pattern}' in {root}"

        result = (
            f"Searched for: '{pattern}' in {root}\n"
            f"Regex: {regex} | Case-sensitive: {case_sensitive} | "
            f"Matches shown: {len(matches)}\n"
            + "-" * 60 + "\n"
        )
        result += "\n".join(matches)

        if len(matches) >= max_matches:
            result += f"\n\n[Reached max_matches limit ({max_matches})]"

        return result

    except Exception as e:
        return f"ERROR searching files: {e}"


def find_files(
    pattern: str,
    path: str = ".",
    max_results: int = 200,
) -> str:
    """
    Find files matching a glob pattern.
    Supports recursive patterns like '**/*.py'.
    """
    root = _resolve(path)

    if not root.exists():
        return f"ERROR: Path not found: {root}"

    if not root.is_dir():
        return f"ERROR: Path is not a directory: {root}"

    try:
        results = []
        for item in sorted(root.glob(pattern)):
            if item.is_file():
                rel = item.relative_to(root)
                size = _format_file_size(item.stat().st_size)
                results.append(f"  FILE  {rel}  ({size})")

                if len(results) >= max_results:
                    break

        # Also include dir matches if they match the pattern
        for item in sorted(root.glob(pattern)):
            if item.is_dir():
                rel = item.relative_to(root)
                results.append(f"  DIR   {rel}")

                if len(results) >= max_results:
                    break

        if not results:
            return f"No files found matching pattern '{pattern}' in {root}"

        header = (
            f"Pattern: '{pattern}' in {root}\n"
            f"Results: {len(results)}"
        )
        if len(results) >= max_results:
            header += f" (limited to {max_results})"

        return header + "\n" + "-" * 60 + "\n" + "\n".join(results)

    except Exception as e:
        return f"ERROR finding files: {e}"


def read_file_lines(
    path: str,
    start_line: int = 1,
    end_line: Optional[int] = None,
    encoding: Optional[str] = None,
) -> str:
    """
    Read a specific section of a file by line numbers.
    Lines are 1-based. Returns the requested lines with line numbers.
    """
    resolved = _resolve(path)

    if not resolved.exists():
        return f"ERROR: File not found: {resolved}"

    if not resolved.is_file():
        return f"ERROR: Path is not a file: {resolved}"

    try:
        content = _read_text(resolved, encoding)
    except Exception as e:
        return f"ERROR reading file: {e}"

    lines = content.splitlines(keepends=False)
    total_lines = len(lines)

    # Validate start_line
    if start_line < 1:
        start_line = 1
    if start_line > total_lines:
        return (
            f"ERROR: start_line ({start_line}) exceeds file length ({total_lines} lines).\n"
            f"File: {resolved} has {total_lines} lines."
        )

    # Default end_line to total_lines if not set
    if end_line is None:
        end_line = total_lines
    if end_line > total_lines:
        end_line = total_lines
    if end_line < start_line:
        return (
            f"ERROR: end_line ({end_line}) is before start_line ({start_line})."
        )

    selected = lines[start_line - 1 : end_line]
    line_count = end_line - start_line + 1

    # Format with line numbers
    digit_width = len(str(end_line))
    formatted = []
    for i, line in enumerate(selected, start=start_line):
        formatted.append(f"{i:>{digit_width}} | {line}")

    result = (
        f"File: {resolved}\n"
        f"Lines: {start_line}–{end_line} of {total_lines} "
        f"({line_count} lines)\n"
        + "-" * 60 + "\n"
    )
    result += "\n".join(formatted)

    return result


def file_diff(
    path_a: str,
    path_b: str,
    context_lines: int = 3,
    encoding: Optional[str] = None,
) -> str:
    """
    Show a unified diff between two files.
    Uses Python's difflib to compute the differences.
    """
    resolved_a = _resolve(path_a)
    resolved_b = _resolve(path_b)

    if not resolved_a.exists():
        return f"ERROR: File not found: {resolved_a}"
    if not resolved_b.exists():
        return f"ERROR: File not found: {resolved_b}"

    if not resolved_a.is_file():
        return f"ERROR: Not a file: {resolved_a}"
    if not resolved_b.is_file():
        return f"ERROR: Not a file: {resolved_b}"

    try:
        text_a = _read_text(resolved_a, encoding)
        text_b = _read_text(resolved_b, encoding)
    except Exception as e:
        return f"ERROR reading files: {e}"

    lines_a = text_a.splitlines(keepends=True)
    lines_b = text_b.splitlines(keepends=True)

    diff = list(
        difflib.unified_diff(
            lines_a,
            lines_b,
            fromfile=str(resolved_a),
            tofile=str(resolved_b),
            n=context_lines,
        )
    )

    if not diff:
        return f"Files are identical: {resolved_a} == {resolved_b}"

    result = (
        f"Diff between:\n"
        f"  A: {resolved_a} ({len(lines_a)} lines)\n"
        f"  B: {resolved_b} ({len(lines_b)} lines)\n"
        + "-" * 60 + "\n"
    )
    result += "".join(diff)

    return result


# =========================================================
# TOOL REGISTRY ENTRIES
# =========================================================

FILE_TOOLS = {
    "read_file": {
        "func": read_file,
        "schema": ReadFileSchema,
        "description": (
            "Read a file from disk and return its text content. "
            "Supports encoding auto-detection and optional truncation."
        ),
    },
    "create_file": {
        "func": create_file,
        "schema": CreateFileSchema,
        "description": (
            "Create a NEW file at the specified path. "
            "By default, this will NOT overwrite an existing file unless overwrite=True. "
            "Use this when you need to create a brand new file. "
            "Parent directories are created automatically."
        ),
    },
    "write_file": {
        "func": write_file,
        "schema": WriteFileSchema,
        "description": (
            "Write or append text content to a file. "
            "Creates parent directories automatically. "
            "Use mode='w' to overwrite or mode='a' to append. "
            "For creating NEW files safely, use create_file instead."
        ),
    },
    "copy_file": {
        "func": copy_file,
        "schema": CopyFileSchema,
        "description": (
            "Copy a file or directory from src to dst. "
            "Supports both files and directories. "
            "Set overwrite=True to replace existing destination."
        ),
    },
    "move_file": {
        "func": move_file,
        "schema": MoveFileSchema,
        "description": (
            "Move or rename a file or directory from src to dst. "
            "Set overwrite=True to replace existing destination."
        ),
    },
    "delete_file": {
        "func": delete_file,
        "schema": DeleteFileSchema,
        "description": (
            "Delete a file at the specified path. "
            "Requires confirm=True as a safety guard against accidental deletion."
        ),
    },
    "create_directory": {
        "func": create_directory,
        "schema": CreateDirectorySchema,
        "description": (
            "Create a directory (and all parent directories). "
            "Works like mkdir -p. "
            "By default does nothing if the directory already exists."
        ),
    },
    "delete_directory": {
        "func": delete_directory,
        "schema": DeleteDirectorySchema,
        "description": (
            "Delete a directory. "
            "Set recursive=True to delete non-empty directories and all contents. "
            "Requires confirm=True as a safety guard."
        ),
    },
    "list_directory": {
        "func": list_directory,
        "schema": ListDirectorySchema,
        "description": (
            "List files and subdirectories at a given path. "
            "Supports recursive listing and extension filtering."
        ),
    },
    "file_info": {
        "func": file_info,
        "schema": FileInfoSchema,
        "description": (
            "Get detailed metadata about a file or directory. "
            "Returns size, modification time, permissions, type, and more."
        ),
    },
    "search_in_files": {
        "func": search_in_files,
        "schema": SearchInFilesSchema,
        "description": (
            "Search for text or regex patterns across multiple files in a directory. "
            "Returns matching lines with file paths and line numbers. "
            "Supports file pattern filtering and case-sensitive/insensitive search."
        ),
    },
    "find_files": {
        "func": find_files,
        "schema": FindFilesSchema,
        "description": (
            "Find files and directories matching a glob pattern. "
            "Supports recursive patterns like '**/*.py'. "
            "Returns relative paths and file sizes."
        ),
    },
    "read_file_lines": {
        "func": read_file_lines,
        "schema": ReadFileLinesSchema,
        "description": (
            "Read a specific line range from a file. "
            "Returns lines with line numbers. "
            "Useful for inspecting specific sections of large files."
        ),
    },
    "file_diff": {
        "func": file_diff,
        "schema": FileDiffSchema,
        "description": (
            "Show a unified diff between two files. "
            "Uses Python's difflib. "
            "Useful for comparing file versions or checking changes."
        ),
    },
}