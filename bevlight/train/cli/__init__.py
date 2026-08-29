"""Command backends: one module per `bevlight train ...` command.

Nothing outside `train/` imports these; the dispatcher reaches them by import
path at run time, and `python -m bevlight.train.cli.<name>` is the same command.
"""
