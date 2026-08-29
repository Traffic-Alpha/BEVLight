'''Every command is a module, and the two ways to run it stay the same thing.

`bevlight eval offline` and `python -m bevlight.eval.cli.offline` are the same
command reached two ways: the first for typing, the second for pdb, cProfile and
IDE launch configs, which want a module rather than a console script. That is
only true while every piece of the mapping holds, and the pieces rot quietly --
before this test the tree had fifteen `tools/` docstrings pointing at module
paths that no longer existed, and seven modules where `python -m` exited 0
having done nothing at all, which is worse than failing.

Nothing here imports a command. The checks read source with `ast`, so they cost
milliseconds and hold even on a machine with neither torch nor SUMO.
'''

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from bevlight.cli.dispatch import command_module, commands_in, groups, summary

REPO = Path(__file__).resolve().parents[1]
PACKAGE = REPO / "bevlight"

COMMANDS = [(group, name) for group in groups() for name in commands_in(group)]
CLI_FILES = sorted(
    path
    for path in PACKAGE.glob("*/cli/*.py")
    if path.name != "__init__.py"
)


def top_level(path: Path) -> dict[str, ast.stmt]:
    return {
        node.name: node
        for node in ast.parse(path.read_text()).body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }


def has_main_guard(path: Path) -> bool:
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.If) and "__main__" in ast.dump(node.test):
            return True
    return False


def test_the_tree_offers_at_least_the_commands_it_used_to():
    """A structural test that passes on an empty tree is not a test."""
    assert len(COMMANDS) >= 20, f"only found {len(COMMANDS)} commands"


@pytest.mark.parametrize("group,name", COMMANDS, ids=lambda v: str(v))
def test_a_command_names_a_module_that_exists_and_can_be_run(group, name):
    """`bevlight <group> <command>` resolves to a file with a callable main.

    The dispatcher builds the import path from the command name rather than
    reading it out of a table, so this is the check that the convention holds
    at every point where it is relied on.
    """
    dotted = command_module(group, name)
    path = REPO / Path(*dotted.split(".")).with_suffix(".py")
    assert path.is_file(), f"{dotted} names no file"
    assert "main" in top_level(path), f"{dotted} defines no main()"


@pytest.mark.parametrize("path", CLI_FILES, ids=lambda p: p.stem)
def test_a_command_module_can_also_be_run_with_dash_m(path):
    """No guard means `python -m` imports the module and exits 0 in silence.

    That failure mode is the reason this file exists: you believe you started a
    training run, the shell agrees, and nothing ran.
    """
    assert has_main_guard(path), (
        f"{path.relative_to(REPO)} has no `if __name__ == \"__main__\"`, so "
        "`python -m` on it would do nothing and report success"
    )


@pytest.mark.parametrize("path", CLI_FILES, ids=lambda p: p.stem)
def test_a_command_says_what_it_does_in_one_line(path):
    """`bevlight --help` is built out of these, so an empty one is a hole in it."""
    assert summary(path), f"{path.relative_to(REPO)} has no @Description line"


@pytest.mark.parametrize("pkg", sorted(
    p.name for p in PACKAGE.iterdir() if (p / "cli").is_dir()
))
def test_the_command_layer_stays_in_the_command_layer(pkg):
    """A library module that grows a `main` has quietly become a command.

    Five of them had: `eval/closed_loop.py`, `eval/offline.py`, `eval/compare.py`,
    `rl/preflight.py` and `rl/sac.py` each carried a `parse_args` and a `main`
    beside their library code, which is how `rl/sac.py` reached 746 lines. The
    fix is to move the command into `<pkg>/cli/`, not to allow it here.
    """
    stray = [
        f"{path.relative_to(REPO)} defines {name}()"
        for path in sorted((PACKAGE / pkg).glob("*.py"))
        for name in ("main", "parse_args")
        if name in top_level(path)
    ]
    assert not stray, "move these into cli/:\n  " + "\n  ".join(stray)
