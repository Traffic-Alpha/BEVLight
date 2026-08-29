# BEVLight

BEVLight studies generalizable traffic signal control from UAV bird's-eye-view
visual observations. A drone hovering over a junction sees every approach at
once, so the control policy reads pixels instead of loop detectors — and it has
to keep working at a junction it has never seen, under a signal plan it has
never seen, from a heading it has never seen.

## Layout

```text
bevlight/          the library; one subpackage per hop of the pipeline
  scenario/        static junction facts: network, BEV camera, lane masks, splits
  expert/          structured state -> phase choice (the expert and the baselines)
  env/             the simulator: SUMO, the renderers, one episode; the observation
                   contract and the reward registry both learners share
  collect/         that loop plus a recorder: labels, Panda images, Blender manifests
  data/            images + trajectory -> padded training samples
  model/           BEV pixels -> lane -> movement -> phase -> decision
  train/           behaviour cloning + auxiliary lane-state regression
  eval/            offline checkpoint scoring, then closed-loop control metrics
  rl/              a different learner on the same world: discrete SAC, reward preflight
  ablation/        the named ablation table and what each row is evidence for
  utils/           paths, TransSimHub/Blender discovery, mask painting

tools/             thin CLIs over the above; no logic lives here
scenarios/         data only: 12 junctions x (networks, routes, add, 3d_assets, lane_mask)
configs/           tracked inputs the code reads: scenario_selection.json, bev_cameras.json
assets/            tracked documentation images, including roadnet previews
data/              generated (gitignored): episodes/, samples/, bev_reference/
runs/              generated (gitignored): backbones/, train/<run>/, reports/
tests/             mask alignment, padding invariance, label consistency, layering
```

Two generated roots rather than one, because they are not equally expensive to
lose: `data/` is hours of Blender, `runs/` is a command rerun. Deleting `runs/`
is always safe; deleting `data/` costs a week.

Inside each subpackage the layout says who may import what. A module at the top
level is other subpackages' business; `_internal/` is that package's alone; and
`cli/` exists to back one `tools/` command. `tests/test_layering.py` fails on a
cross-package import that reaches into either, so the split records a measured
fact rather than an intention.

`scenarios/` holds only per-junction data and the two scripts that generated it
(`config.py`, `generate_routes.py` -- one hand-calibrated pair per junction, kept
beside the routes they produced). `bevlight/` writes nothing into it except the
one-off static artifacts (`3d_assets/`, `lane_mask/`) that are built per junction
and then reused by every episode.

`assets/roadnets/` holds the small tracked visual set used to introduce the 12
experimental roadnets in README/docs. Full frame sequences and logs remain in
gitignored `data/` and `runs/`.

## Setup

TransSimHub is used from a sibling checkout rather than PyPI. Point `TSHUB_ROOT`
at it (or pass `--tshub-root`); `BLENDER` locates the Blender executable. The
SUMO environment comes from the `tshub` conda env, so commands run as
`conda run -n tshub python ...`.

```bash
pip install -e .          # optional; tools/ scripts work without it
```

## Scenarios

The full pool is 120 combinations (12 junctions x 2 signal plans x 5 demands).
Reported experiments use the 64-scenario active subset in
[`configs/scenario_selection.json`](configs/scenario_selection.json),
split into 48 training, 8 cross-plan test and 8 cross-structure test scenarios.
Load it through `bevlight.scenario.selection`, never by scanning `*.sumocfg` —
see [`docs/2-scenarios.md`](docs/2-scenarios.md).

The two plans differ in phase composition, not just timing: `easy` releases one
approach at a time, `normal` pairs opposing movements, and at `Hongkong_YMT` they
differ in phase count as well (4 vs 3). That is what the cross-plan axis tests.

## Pipeline

Each stage is one command; every stage after the first consumes only what the
previous one wrote.

```bash
# One-off per junction: networks, 3D assets, lane masks.
conda run -n tshub python tools/build_networks.py
conda run -n tshub python tools/build_static_scenes.py
conda run -n tshub python tools/render_bev_reference.py
conda run -n tshub python tools/build_lane_masks.py

# Inspect what the model sees: one image per lane.
conda run -n tshub python tools/export_lane_views.py --junction Beijing_Beihuan

# Check the expert is worth imitating, then collect expert episodes.
# Collection runs SUMO once, records actions/labels, and renders all Panda RGB/SEG frames.
conda run -n tshub python tools/compare_experts.py --junction Beijing_Beihuan
conda run -n tshub python tools/collect_episodes.py --junction Beijing_Beihuan

# Render selected Blender frames, then check the model can read them.
conda run -n tshub python tools/download_backbone.py
conda run -n tshub python tools/render_blender.py --episode-dir data/episodes/<key> --passes rgb
conda run -n tshub python tools/flatten_blender_images.py --episode-dir data/episodes/<key> --passes rgb
conda run -n tshub python tools/build_dataset.py --name pinganli_pilot --junction Beijing_Pinganli
conda run -n tshub python tools/probe_queue.py --dataset pinganli_pilot

# Behaviour-clone the expert on the cached features.
conda run -n tshub python tools/train.py --dataset pinganli_pilot --run pinganli_bc

# Score the checkpoints offline (seconds), then rank the shortlist in closed loop.
conda run -n tshub python tools/eval_offline.py --run pinganli_bc
conda run -n tshub python tools/eval_closed_loop.py --run pinganli_bc --split train \
    --junction Beijing_Pinganli --baseline fixed_time max_pressure

# Check a demand lands queues in the observable band (the answer depends on the controller).
conda run -n tshub python tools/calibrate_demand.py --junction Beijing_Pinganli \
    --scale 1.0 1.2 --controller max_pressure fixed_time
```

Every tool takes `--dry-run` and prints its planned jobs before touching
anything.

## Status

**A policy reading only BEV pixels matches max-pressure, which reads structured
state no real deployment can obtain — and it holds at a junction it has never
seen.** Over 91 scenarios and 273 closed-loop episodes: travel time −24.2% and
waiting −58.2% against fixed time, against max-pressure's −23.9% and −58.4%,
with identical throughput in 88 of 91.

The three generalization axes, as median travel time against the expert:

| split | scenarios | median Δ | vs fixed time | failures |
|---|---:|---:|---:|---:|
| train (seen) | 45 | +0.00 s | +10.9 s | 0 |
| cross-demand | 30 | −0.12 s | +12.1 s | 0 |
| cross-plan | 8 | +0.20 s | +14.5 s | 0 |
| cross-structure | 8 | +0.48 s | +13.4 s | 0 |

Degradation across the axes is monotone and small, every value inside the
measurement's own 1.2 s noise floor. Nothing failed anywhere; the worst single
scenario of 91 is 6.4 s behind the expert and still ahead of fixed time.
SouthKorea_Songdo carries **40 lanes against a training maximum of 28** and is
never touched by training.

Behind that: 45 episodes across ten junctions replayed into four appearances for
10 288 decision samples; one shared probe head reads queue length across all
fifteen (junction, plan) groups at R² 0.769, every group between 0.768 and 0.850.

**One run per scenario, one seed.** Every median above is smaller than the noise
floor, so the supported claim is *no detectable gap to the expert*, never *equal
to* it. Repeats, a one-hot control and the ablations are unrun. Every number,
including what has **not** been measured, is in
[`docs/9-results.md`](docs/9-results.md).

## Docs

Start at [`docs/README.md`](docs/README.md) — the pipeline map and the reading
order. Every experimental number is consolidated in
[`docs/9-results.md`](docs/9-results.md); the other documents describe how each
stage works.
