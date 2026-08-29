'''The front door: `bevlight <group> <command>`, plus what those commands share.

`dispatch` is the entry point `pyproject.toml` installs as `bevlight`. The other
modules here are what more than one package's `cli/` needs and no library module
does -- locating TransSimHub, painting a mask for a human to look at. They live
outside the pipeline packages because they belong to none of them.
'''

from .dispatch import main

__all__ = ["main"]
