'''Who may import whom. The rule the directory layout encodes, enforced.

Every subpackage is three tiers -- the public surface at its top level,
`_internal/` for what only it uses, `cli/` for the commands it exposes. That
split records a measured fact: which modules actually have callers outside
their own package. Documentation of such a fact rots; this does not.

Imports are read with `ast` rather than by importing anything, so the check
costs milliseconds and needs neither torch nor SUMO.
'''

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
PACKAGE = REPO / "bevlight"

PRIVATE_TIERS = ("_internal", "cli")


def python_files(root: Path) -> list[Path]:
    return sorted(
        f for f in root.rglob("*.py") if "__pycache__" not in f.parts
    )


def module_name(path: Path) -> str:
    parts = list(path.relative_to(REPO).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def package_of(path: Path) -> str:
    """The dotted package a file's relative imports resolve against."""
    parts = list(path.relative_to(REPO).with_suffix("").parts)
    parts.pop()
    return ".".join(parts)


def resolve(pkg: str, level: int, module: str | None) -> str | None:
    """A `from ... import` target as an absolute dotted path."""
    if level == 0:
        return module
    parts = pkg.split(".")
    keep = len(parts) - (level - 1)
    if keep < 1:
        return None
    return ".".join(parts[:keep] + (module.split(".") if module else []))


def imports_of(path: Path) -> list[tuple[int, str]]:
    """Every `bevlight.*` target this file imports, as (line, dotted path)."""
    pkg = package_of(path)
    found = []
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.ImportFrom):
            target = resolve(pkg, node.level, node.module)
            if not target or not target.startswith("bevlight"):
                continue
            # `from bevlight.scenario import lane_mask` names the module in the
            # import list, not the module path.
            for alias in node.names:
                found.append((node.lineno, f"{target}.{alias.name}"))
            found.append((node.lineno, target))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("bevlight"):
                    found.append((node.lineno, alias.name))
    return found


def subpackage(dotted: str) -> str | None:
    """`bevlight.model._internal.phase` -> `model`."""
    parts = dotted.split(".")
    return parts[1] if len(parts) > 1 and parts[0] == "bevlight" else None


def tier(dotted: str) -> str | None:
    parts = dotted.split(".")
    for part in parts[2:]:
        if part in PRIVATE_TIERS:
            return part
    return None

ALL_MODULES = {module_name(f) for f in python_files(PACKAGE)}


def test_no_subpackage_reaches_into_another_subpackages_internals():
    """`_internal/` and `cli/` are private to their own subpackage.

    This is the whole rule. A break here means a module that was classified as
    "nobody else's business" acquired an outside caller -- which is a real
    change in the design, and the fix is to move the file up a tier rather than
    to loosen this test.
    """
    breaks = []
    for path in python_files(PACKAGE):
        importer = subpackage(module_name(path))
        for line, target in imports_of(path):
            if target not in ALL_MODULES:
                continue
            hit = tier(target)
            if hit and subpackage(target) != importer:
                breaks.append(
                    f"{path.relative_to(REPO)}:{line} -> {target} "
                    f"(reaches {subpackage(target)}/{hit}/ from {importer}/)"
                )
    assert not breaks, "layering breaks:\n  " + "\n  ".join(breaks)


def test_nothing_imports_the_front_door():
    """The dispatcher depends on the packages. Nothing may depend on it back.

    `bevlight/cli/dispatch.py` resolves a command name to an import path at run
    time, so a package that imported it would close a loop the import graph does
    not otherwise have -- and would make `--help` cost whatever that package
    costs. Its two neighbours, `cli/tshub.py` and `cli/viz.py`, are ordinary
    shared modules and are not covered by this.
    """
    allowed = {"bevlight.cli", "bevlight.cli.dispatch"}
    breaks = [
        f"{path.relative_to(REPO)}:{line}"
        for path in python_files(REPO / "bevlight") + python_files(REPO / "tests")
        if module_name(path) not in allowed
        for line, target in imports_of(path)
        if target.startswith("bevlight.cli.dispatch")
        and not module_name(path).startswith("tests.")
    ]
    assert not breaks, "importing the dispatcher:\n  " + "\n  ".join(breaks)


def callers() -> dict[str, set[str]]:
    """For each module, which other modules import it."""
    users: dict[str, set[str]] = {m: set() for m in ALL_MODULES}
    for root in ("bevlight", "tests"):
        for path in python_files(REPO / root):
            here = module_name(path)
            for _, target in imports_of(path):
                if target in users and target != here:
                    users[target].add(here)
    return users


def reexported(pkg: str) -> set[str]:
    """Modules a subpackage's `__init__` pulls names out of."""
    init = PACKAGE / pkg / "__init__.py"
    if not init.is_file():
        return set()
    return {t for _, t in imports_of(init) if t in ALL_MODULES}


@pytest.mark.parametrize("pkg", sorted(
    p.name for p in PACKAGE.iterdir() if (p / "__init__.py").is_file()
))
def test_no_module_is_left_with_nobody_calling_it(pkg):
    """A module nothing imports is dead weight, and reads as live code.

    This used to demand a caller *outside* the package, which made sense while
    every command lived in `tools/` and therefore outside. Now that a command is
    `<pkg>/cli/<name>.py`, a module whose only caller is its own package's
    command is doing exactly what it should -- `rl/preflight.py` backs
    `bevlight rl preflight` and nothing else, and that is not a defect. So the
    check is the weaker, still-useful one: somebody has to call it.
    """
    users = callers()
    exported = reexported(pkg)
    stranded = []
    for path in sorted((PACKAGE / pkg).glob("*.py")):
        name = module_name(path)
        if path.name == "__init__.py" or name in exported:
            continue
        if not users.get(name):
            stranded.append(f"{path.relative_to(REPO)} is imported by nothing")
    assert not stranded, (
        "delete these, or export them from __init__ if they are the surface:\n  "
        + "\n  ".join(stranded)
    )


@pytest.mark.parametrize("pkg", sorted(
    p.name for p in PACKAGE.iterdir() if (p / "__init__.py").is_file()
))
def test_declared_public_names_resolve(pkg):
    """`__all__` is the advertised surface, so it has to actually be there."""
    import importlib

    module = importlib.import_module(f"bevlight.{pkg}")
    missing = [n for n in getattr(module, "__all__", []) if not hasattr(module, n)]
    assert not missing, f"bevlight.{pkg}.__all__ names nothing: {missing}"
