"""Command backends: one module per `bevlight collect ...` command.

Nothing outside `collect/` imports these; the dispatcher reaches them by import
path at run time, and `python -m bevlight.collect.cli.<name>` is the same command.
"""
