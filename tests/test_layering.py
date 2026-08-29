'''Who may import whom. The rule the directory layout encodes, enforced.

Every subpackage is three tiers -- the public surface at its top level,
`_internal/` for what only it uses, `cli/` for what only a `tools/` command
uses. That split records a measured fact: which modules actually have callers
outside their own package. Documentation of such a fact rots; this does not.

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


def test_tools_use_cli_backends_and_never_package_internals():
    """`tools/` is the intended caller of `cli/`, and of nothing private."""
    breaks = []
    for path in python_files(REPO / "tools"):
        for line, target in imports_of(path):
            if target in ALL_MODULES and tier(target) == "_internal":
                breaks.append(f"{path.relative_to(REPO)}:{line} -> {target}")
    assert not breaks, "tools/ reaching into _internal/:\n  " + "\n  ".join(breaks)


def external_users() -> dict[str, set[str]]:
    """For each module, which places outside its own subpackage import it."""
    users: dict[str, set[str]] = {m: set() for m in ALL_MODULES}
    for root in ("bevlight", "tools", "tests"):
        for path in python_files(REPO / root):
            here = subpackage(module_name(path)) if root == "bevlight" else root
            for _, target in imports_of(path):
                if target in users and subpackage(target) != here:
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
def test_public_modules_still_have_outside_callers(pkg):
    """A top-level module claims to be other people's business. Check it is.

    This is the drift check in the other direction: a module whose last outside
    caller went away is now internal in fact while still sitting in public
    space, and the next reader has no way to tell. Either it moves into
    `_internal/`, or `__init__` exports it as the package's surface.
    """
    users = external_users()
    exported = reexported(pkg)
    stranded = []
    for path in sorted((PACKAGE / pkg).glob("*.py")):
        name = module_name(path)
        if path.name == "__init__.py" or name in exported:
            continue
        if not users.get(name):
            stranded.append(f"{path.relative_to(REPO)} has no caller outside {pkg}/")
    assert not stranded, (
        "move these into _internal/, or export them from __init__:\n  "
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
