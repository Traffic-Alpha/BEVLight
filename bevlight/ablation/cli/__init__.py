"""Command backends: one module per `bevlight ablation ...` command.

Nothing outside `ablation/` imports these; the dispatcher reaches them by import
path at run time, and `python -m bevlight.ablation.cli.<name>` is the same command.
"""
