"""Command backends: one module per `bevlight model ...` command.

Nothing outside `model/` imports these; the dispatcher reaches them by import
path at run time, and `python -m bevlight.model.cli.<name>` is the same command.
"""
