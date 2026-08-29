"""Command backends: one module per `bevlight eval ...` command.

Nothing outside `eval/` imports these; the dispatcher reaches them by import
path at run time, and `python -m bevlight.eval.cli.<name>` is the same command.
"""
