'''`bevlight <group> <command>` -- the package layout, read back as a CLI.

There is no registry table. A group is a pipeline package, a command is a module
in that package's `cli/`, and the mapping is the import path itself:

    bevlight eval offline  ->  bevlight.eval.cli.offline:main
                           ->  python -m bevlight.eval.cli.offline

Both forms are supported and must stay equivalent: the first is for typing, the
second is for pdb, cProfile and IDE launch configs, which want a module rather
than a console script. `tests/test_cli_mapping.py` fails if they drift apart.

Two things keep `bevlight --help` cheap. Commands are discovered by listing
directories rather than importing packages, and each summary line is read out of
the module's docstring with `ast`, so printing the help does not import torch,
TransSimHub or SUMO -- only the one command you actually run gets imported.
'''

from __future__ import annotations

import ast
import importlib
import sys
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent.parent
THIS_GROUP = Path(__file__).resolve().parent.name

# The order commands run in, which is not alphabetical order. A newcomer reading
# `bevlight --help` should see the pipeline, not the filesystem.
GROUP_ORDER = [
    "scenario",
    "collect",
    "data",
    "model",
    "train",
    "eval",
    "rl",
    "ablation",
]


def command_module(group: str, command: str) -> str:
    """`("eval", "closed-loop")` -> `"bevlight.eval.cli.closed_loop"`."""
    return f"bevlight.{group}.cli.{command.replace('-', '_')}"


def summary(path: Path) -> str:
    """The command's one-line description, without importing it."""
    try:
        doc = ast.get_docstring(ast.parse(path.read_text())) or ""
    except (OSError, SyntaxError):
        return ""
    for line in doc.splitlines():
        line = line.strip()
        if line.startswith("@Description:"):
            return line.removeprefix("@Description:").strip()
    for line in doc.splitlines():
        if line.strip() and not line.strip().startswith("@"):
            return line.strip()
    return ""


def commands_in(group: str) -> dict[str, Path]:
    """Every command a group offers, as `name -> module file`."""
    cli_dir = PACKAGE / group / "cli"
    if not cli_dir.is_dir():
        return {}
    return {
        path.stem.replace("_", "-"): path
        for path in sorted(cli_dir.glob("*.py"))
        if path.name != "__init__.py"
    }


def groups() -> list[str]:
    """Pipeline packages that ship commands, in pipeline order."""
    found = {
        path.name
        for path in PACKAGE.iterdir()
        if path.is_dir() and path.name != THIS_GROUP and (path / "cli").is_dir()
    }
    ordered = [g for g in GROUP_ORDER if g in found]
    return ordered + sorted(found - set(ordered))


def format_help() -> str:
    lines = [
        "usage: bevlight <group> <command> [options]",
        "",
        "Commands, in the order the pipeline runs them.",
        "",
    ]
    for group in groups():
        found = commands_in(group)
        if not found:
            continue
        lines.append(f"{group}:")
        width = max(len(name) for name in found)
        for name, path in found.items():
            lines.append(f"  {name:<{width}}  {summary(path)}")
        lines.append("")
    lines += [
        "Every command is also a module, and the two are equivalent:",
        "",
        "  bevlight eval offline --run baseline",
        "  python -m bevlight.eval.cli.offline --run baseline",
        "",
        "The second form is the one to reach for under pdb or a profiler.",
        "Run `bevlight <group>` to list one group, or `bevlight <group> <command>",
        "--help` for a command's own options.",
    ]
    return "\n".join(lines)


def format_group_help(group: str) -> str:
    found = commands_in(group)
    width = max((len(name) for name in found), default=0)
    lines = [f"usage: bevlight {group} <command> [options]", ""]
    for name, path in found.items():
        lines.append(f"  {name:<{width}}  {summary(path)}")
    return "\n".join(lines)


def unknown(what: str, given: str, known: list[str]) -> int:
    import difflib

    print(f"bevlight: unknown {what} {given!r}", file=sys.stderr)
    close = difflib.get_close_matches(given, known, n=3)
    if close:
        print(f"did you mean: {', '.join(close)}?", file=sys.stderr)
    else:
        print(f"{what}s: {', '.join(known)}", file=sys.stderr)
    return 2


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    if not argv or argv[0] in ("-h", "--help", "help"):
        print(format_help())
        return 0
    if argv[0] in ("-V", "--version"):
        from importlib.metadata import version

        print(version("bevlight"))
        return 0

    group, rest = argv[0], argv[1:]
    if group not in groups():
        return unknown("group", group, groups())

    found = commands_in(group)
    if not rest or rest[0] in ("-h", "--help"):
        print(format_group_help(group))
        return 0

    command, args = rest[0], rest[1:]
    if command not in found:
        return unknown("command", command, list(found))

    # The command parses `sys.argv` itself, exactly as it does under `-m`, so
    # every `main()` signature in the tree is called the same way.
    module = importlib.import_module(command_module(group, command))
    sys.argv = [f"bevlight {group} {command}", *args]
    return module.main() or 0


if __name__ == "__main__":
    raise SystemExit(main())
