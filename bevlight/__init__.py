'''BEVLight: generalizable traffic signal control from UAV bird's-eye-view video.

The pipeline runs in one direction, and each subpackage owns exactly one hop:

    scenario  static junction facts (network, lane masks, BEV camera, splits)
        |
    expert    structured state -> phase choice (also the baselines)
        |
    collect   SUMO rollout -> labels + Panda images + Blender manifests
        |
    data      images + trajectory -> padded training samples
        |
    model     BEV pixels -> lane -> movement -> phase -> decision
        |
    train     behaviour cloning + auxiliary lane-state regression
        |
    eval      offline checkpoint scoring, then closed-loop control metrics

Two subpackages sit beside that line rather than on it:

    env       the world itself: SUMO, the renderers, one episode. `collect`,
              `eval` and `rl` are three things to do with that one loop
    rl        a different learner on that world, above `env` and never below
              it -- nothing on the pipeline above imports `rl`

`tools/` holds thin CLIs over these; `scenarios/` holds data only.

## Where a file sits says who may import it

Every subpackage is three tiers, so that reading a directory listing answers
"is this someone else's business" without grepping:

    <pkg>/__init__.py     the public surface: re-exports, with __all__
    <pkg>/*.py            implementation other subpackages legitimately use
    <pkg>/_internal/*.py  only <pkg> itself may import these
    <pkg>/cli/*.py        exists to back one tools/ command; only tools/ imports it

A cross-subpackage import that reaches into `_internal/` or `cli/` is a layering
break, and `tests/test_layering.py` fails on it. `tools/` and `tests/` are the
two exceptions -- they are the intended callers of `cli/`, and a test may reach
anywhere it needs to.

The tiers are not a style preference. They record a fact that was measured:
which modules actually have callers outside their own package. When that fact
changes, the file moves.
'''

__version__ = "0.1.0"
