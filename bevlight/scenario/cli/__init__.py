"""Command backends: one module per `bevlight scenario ...` command.

Nothing outside `scenario/` imports these; the dispatcher reaches them by import
path at run time, and `python -m bevlight.scenario.cli.<name>` is the same command.
"""
