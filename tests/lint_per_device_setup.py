#!/usr/bin/env python3
"""Rejects kernel setup performed by a plain `static` initializer under src/ops.

cudaFuncSetAttribute and cudaFuncSetCacheConfig apply to the current device, so a `static`
initialized once per process configures only the first device a launcher runs on. Launchers go
through PerDeviceOnce (src/core/per_device.h) instead. Usage: lint_per_device_setup.py <src/ops>
"""
import pathlib
import re
import sys

SETUP = re.compile(r"\bcudaFuncSet(?:Attribute|CacheConfig)\b")
COMMENT = re.compile(r"//[^\n]*|/\*.*?\*/", re.S)
STATIC = re.compile(r"\bstatic\s")
SOURCES = {".cu", ".cuh", ".cpp", ".h"}


def initialized_declaration(text, start):
    """Text of the variable declaration starting at `start`, or None for anything else."""
    depth, assigned = 0, False
    for index in range(start, len(text)):
        char = text[index]
        if depth == 0 and not assigned:
            if char == "=":
                assigned = True
            elif char in "({;":
                return None  # function, direct-initialized, or uninitialized declaration
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
            if depth < 0:
                return None
        elif char == ";" and depth == 0:
            return text[start:index]
    return None


def violations(path):
    text = COMMENT.sub(lambda match: "\n" * match.group().count("\n"), path.read_text())
    for match in STATIC.finditer(text):
        declaration = initialized_declaration(text, match.end())
        if declaration and SETUP.search(declaration) and "PerDeviceOnce" not in declaration:
            yield text.count("\n", 0, match.start()) + 1


def main():
    root = pathlib.Path(sys.argv[1])
    found = [(path, line) for path in sorted(root.rglob("*")) if path.suffix in SOURCES
             for line in violations(path)]
    for path, line in found:
        print(f"{path}:{line}: kernel setup in a plain static initializer; use PerDeviceOnce")
    print(f"{len(found)} violation(s)" if found else "ok")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
