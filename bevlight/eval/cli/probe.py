'''
@Author: WANG Maonan
@Date: 2026-08-21
@Description: Can BEV + lane mask recover queue length at all?

This settles the question the rest of the design rests on, before any decision
head exists. The claim being tested is narrow and checkable: a frozen DINOv2
feature map, pooled through a static lane mask, contains enough to read how many
vehicles are queued on each incoming lane.

If it does not, nothing downstream can work, and the fault is in the perception
design (window, pixel scale, patch coverage, pooling) rather than in the control
head — which is exactly why it is worth asking first and separately.

Three references make the numbers mean something:

  * predict zero          - what a model that ignores the image achieves
  * predict the mean      - what a model that learns only the prior achieves
  * predict per-lane mean - what a model that memorises which lane is busy gets,
                            without reading the image at all

A probe only beats the last one by actually looking at the picture.

Two traps this had to be built around. Queue counts are sparse — most lanes are
empty most of the time — so an L1 objective is minimised by predicting the
median, which is zero, and a collapsed head then scores an *excellent* MAE while
having learned nothing. The loss therefore defaults to MSE, and metrics are
reported a second time over the non-empty lanes only, where a collapsed head has
nowhere to hide. For the same reason `vehicles` (every vehicle on the visible
stretch, moving or not) is available as a target: it is dense, so it separates
"the features cannot see cars" from "the target was too sparse to fit".

Two splits are reported. A random frame split asks whether the information is
present. A leave-one-demand-out split asks whether it survives a traffic regime
the head never saw, which is the property that matters later.
@LastEditTime: 2026-08-21
'''

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ...utils.paths import REPORTS_ROOT, SAMPLES_ROOT


def load_cache(dataset_name: str) -> dict:
    path = SAMPLES_ROOT / dataset_name / "lane_features.npz"
    if not path.is_file():
        raise SystemExit(
            f"No feature cache at {path}. Build one with tools/build_dataset.py."
        )
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def lane_groups(cache: dict, episodes_meta: dict, junctions=None, plans=None) -> list[dict]:
    """One group per (junction, plan): its frames and its incoming-lane columns.

    A cache can hold ten junctions with 12 to 28 lanes each, so "column 5" means a
    different lane in each of them. Every group therefore carries its own columns,
    and lanes are named globally so a per-lane baseline cannot pool two junctions'
    lane 5 into one average.
    """
    lane_orders = json.loads(str(cache["lane_orders"]))
    roles_of = {episode: meta["lane_roles"] for episode, meta in episodes_meta.items()}

    groups: dict[tuple, dict] = {}
    for row, (episode, junction, plan) in enumerate(
        zip(cache["episode"], cache["junction"], cache["plan"])
    ):
        episode, junction, plan = str(episode), str(junction), str(plan)
        if junctions and junction not in junctions:
            continue
        if plans and plan not in plans:
            continue
        key = (junction, plan)
        if key not in groups:
            order = lane_orders[episode]
            roles = roles_of[episode]
            columns = [i for i, lane in enumerate(order) if roles[lane] == "incoming"]
            groups[key] = {
                "key": f"{junction}/{plan}",
                "columns": np.array(columns, dtype=np.int64),
                "lane_ids": [f"{junction}/{order[i]}" for i in columns],
                "rows": [],
            }
        groups[key]["rows"].append(row)

    ordered = []
    for offset, key in enumerate(sorted(groups)):
        group = groups[key]
        group["rows"] = np.array(group["rows"], dtype=np.int64)
        # Globally unique lane ids, so `predict_lane_mean` averages per real lane.
        group["lane_base"] = offset * 1000
        ordered.append(group)
    return ordered


def metrics(prediction: np.ndarray, target: np.ndarray) -> dict:
    error = prediction - target
    ss_res = float((error ** 2).sum())
    ss_tot = float(((target - target.mean()) ** 2).sum())
    nonempty = target > 0
    return {
        "mae": round(float(np.abs(error).mean()), 4),
        "rmse": round(float(np.sqrt((error ** 2).mean())), 4),
        "r2": round(1.0 - ss_res / ss_tot, 4) if ss_tot > 0 else float("nan"),
        "within_1_veh": round(float((np.abs(error) <= 1.0).mean()), 4),
        # Where a collapsed head cannot hide: lanes that actually hold traffic.
        "mae_nonempty": round(float(np.abs(error[nonempty]).mean()), 4) if nonempty.any() else 0.0,
        "n_nonempty": int(nonempty.sum()),
        "n": int(target.size),
    }


def train_probe(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    epochs: int = 300,
    hidden: int = 256,
    lr: float = 1e-3,
    depth: int = 2,
    device: str | None = None,
    seed: int = 0,
    loss: str = "mse",
) -> tuple[dict, np.ndarray]:
    """Fit the shared per-lane head on cached features."""
    from ...model.heads import QueueHead

    torch.manual_seed(seed)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    xt = torch.from_numpy(train_x).float().to(dev)
    yt = torch.from_numpy(train_y).float().to(dev)
    xv = torch.from_numpy(test_x).float().to(dev)

    head = QueueHead(train_x.shape[-1], hidden=hidden, depth=depth).to(dev)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=0.01)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # MSE by default: with a sparse target the L1 optimum is the median, which is
    # zero, so an L1 head collapses to predicting nothing and still scores well.
    criterion = {
        "mse": torch.nn.functional.mse_loss,
        "l1": torch.nn.functional.l1_loss,
        "huber": torch.nn.functional.smooth_l1_loss,
    }[loss]

    # The head is shared across lanes, so a "batch" here is just lane vectors.
    for _ in range(epochs):
        head.train()
        optimizer.zero_grad()
        objective = criterion(head(xt.unsqueeze(0)).squeeze(0), yt)
        objective.backward()
        optimizer.step()
        schedule.step()

    head.eval()
    with torch.no_grad():
        prediction = head(xv.unsqueeze(0)).squeeze(0).cpu().numpy()
    return metrics(prediction, test_y), prediction


def baselines(train_y: np.ndarray, test_y: np.ndarray, lane_of: np.ndarray,
              train_lane: np.ndarray) -> dict:
    """What you get without reading the image."""
    zero = np.zeros_like(test_y)
    mean = np.full_like(test_y, train_y.mean())

    per_lane = np.zeros_like(test_y)
    for lane in np.unique(lane_of):
        subset = train_y[train_lane == lane]
        per_lane[lane_of == lane] = subset.mean() if subset.size else train_y.mean()

    return {
        "predict_zero": metrics(zero, test_y),
        "predict_mean": metrics(mean, test_y),
        "predict_lane_mean": metrics(per_lane, test_y),
    }


def flatten(cache: dict, groups: list, rows: np.ndarray, drop_saturated: bool,
            target: str = "queued"):
    """(frames, lanes, D) -> (frames*lanes, D), with the matching targets.

    Rows are selected across every group, so one shared head is fitted over all
    junctions at once — which is the claim being tested, not "a head per junction".
    """
    wanted = np.zeros(cache["features"].shape[0], dtype=bool)
    wanted[rows] = True

    xs, ys, lanes, owners = [], [], [], []
    for position, group in enumerate(groups):
        picked = group["rows"][wanted[group["rows"]]]
        if not picked.size:
            continue
        columns = group["columns"]
        features = cache["features"][picked][:, columns, :].astype(np.float32)
        values = cache[target][picked][:, columns].astype(np.float32)
        saturated = cache["saturated"][picked][:, columns]
        lane_of = np.broadcast_to(
            group["lane_base"] + np.arange(columns.size), values.shape
        )

        keep = ~saturated.reshape(-1) if drop_saturated else np.ones(values.size, dtype=bool)
        xs.append(features.reshape(-1, features.shape[-1])[keep])
        ys.append(values.reshape(-1)[keep])
        lanes.append(lane_of.reshape(-1)[keep])
        # Which group each lane sample came from, so a pooled fit can still be
        # read per junction afterwards.
        owners.append(np.full(int(keep.sum()), position, dtype=np.int64))

    if not xs:
        empty_x = np.zeros((0, cache["features"].shape[-1]), dtype=np.float32)
        empty = np.zeros(0, dtype=np.int64)
        return empty_x, np.zeros(0, dtype=np.float32), empty, empty
    return (np.concatenate(xs), np.concatenate(ys),
            np.concatenate(lanes), np.concatenate(owners))


def shared_head_by_group(prediction: np.ndarray, target: np.ndarray,
                         owner: np.ndarray, groups: list) -> dict:
    """The *pooled* head's score at each junction.

    Different question from fitting a head per junction. That one asks whether the
    information is present somewhere in a junction's features; this one asks
    whether one shared set of weights reads all of them — which is the claim, and
    the thing a pooled average is most able to hide.
    """
    return {
        groups[position]["key"]: metrics(prediction[owner == position],
                                        target[owner == position])
        for position in range(len(groups))
        if (owner == position).any()
    }


def report_by_group(block) -> None:
    if not block:
        return
    print(f"    {'the shared head, per junction':<32}{'MAE':>8}{'R2':>8}{'MAE>0':>8}{'n':>9}")
    for key, m in sorted(block.items(), key=lambda kv: kv[1]["r2"]):
        print(f"    {key:<32}{m['mae']:8.3f}{m['r2']:8.3f}{m['mae_nonempty']:8.3f}{m['n']:9d}")


def probe_splits(cache: dict, groups: list, args, label: str = "") -> dict:
    """Both splits, over whatever set of groups is handed in."""
    rows = np.concatenate([g["rows"] for g in groups])
    demands = cache["demand"]
    results = {}
    rng = np.random.default_rng(args.seed)

    # --- split 1: random frames. Is the information there at all? ---
    order = rng.permutation(rows)
    cut = int(order.size * (1 - args.test_fraction))
    xt, yt, lt, _ = flatten(cache, groups, order[:cut], args.drop_saturated, args.target)
    xv, yv, lv, gv = flatten(cache, groups, order[cut:], args.drop_saturated, args.target)
    print(f"\n=== {label}random frame split ===  train={yt.size} lane-samples  test={yv.size}")
    probe, prediction = train_probe(xt, yt, xv, yv, epochs=args.epochs, hidden=args.hidden,
                                    depth=args.depth, lr=args.lr, seed=args.seed, loss=args.loss)
    results["random_split"] = {"probe": probe, **baselines(yt, yv, lv, lt)}
    if len(groups) > 1:
        results["random_split"]["by_group"] = shared_head_by_group(
            prediction, yv, gv, groups
        )
    report(results["random_split"])
    report_by_group(results["random_split"].get("by_group"))

    # --- split 2: leave one demand out. Does it survive a new traffic regime? ---
    for demand in sorted({str(d) for d in demands[rows].tolist()}):
        is_demand = demands[rows] == demand
        train_rows, test_rows = rows[~is_demand], rows[is_demand]
        if not train_rows.size or not test_rows.size:
            continue
        xt, yt, lt, _ = flatten(cache, groups, train_rows, args.drop_saturated, args.target)
        xv, yv, lv, gv = flatten(cache, groups, test_rows, args.drop_saturated, args.target)
        print(f"\n=== {label}held-out demand: {demand} ===  train={yt.size}  test={yv.size}")
        probe, prediction = train_probe(xt, yt, xv, yv, epochs=args.epochs, hidden=args.hidden,
                                        depth=args.depth, lr=args.lr, seed=args.seed, loss=args.loss)
        results[f"heldout_{demand}"] = {"probe": probe, **baselines(yt, yv, lv, lt)}
        if len(groups) > 1:
            results[f"heldout_{demand}"]["by_group"] = shared_head_by_group(
                prediction, yv, gv, groups
            )
        report(results[f"heldout_{demand}"])
        report_by_group(results[f"heldout_{demand}"].get("by_group"))

    return results


def run(args) -> dict:
    cache = load_cache(args.dataset)
    index_meta = json.loads((SAMPLES_ROOT / args.dataset / "index.json").read_text())
    groups = lane_groups(cache, index_meta["episodes"], args.junction, args.plan)
    if not groups:
        raise SystemExit("No frames matched --junction / --plan.")

    lanes = sum(g["columns"].size for g in groups)
    frames = sum(g["rows"].size for g in groups)
    print(
        f"[probe] dataset={args.dataset} backbone={cache['backbone']} "
        f"frames={frames} groups={len(groups)} incoming_lanes={lanes} "
        f"dim={cache['features'].shape[-1]}"
    )
    for group in groups:
        print(f"        {group['key']:<32} frames={group['rows'].size:<6} "
              f"incoming_lanes={group['columns'].size}")

    _, values, _, _ = flatten(cache, groups, np.concatenate([g["rows"] for g in groups]),
                              False, args.target)
    print(f"        target={args.target}: mean={values.mean():.2f} max={values.max():.0f} "
          f"zeros={100*(values == 0).mean():.1f}%")
    if (values == 0).mean() > 0.9:
        print("        note: target is >90% zero; watch mae_nonempty, not mae")

    # One head over every junction at once. That is the claim; a head per
    # junction would only say the features are readable somewhere.
    results = {"pooled": probe_splits(cache, groups, args)}

    if len(groups) > 1 and not args.pooled_only:
        # Per junction as well, because a pooled average can hide one junction
        # the probe cannot read at all.
        results["per_group"] = {}
        for group in groups:
            print(f"\n########## {group['key']} ##########")
            results["per_group"][group["key"]] = probe_splits(
                cache, [group], args, label=f"{group['key']} "
            )
    return results


def report(block: dict) -> None:
    order = ["probe", "predict_lane_mean", "predict_mean", "predict_zero"]
    print(f"    {'method':20s} {'MAE':>8s} {'RMSE':>8s} {'R2':>8s} {'MAE>0':>8s} {'<=1 veh':>9s}")
    for name in order:
        m = block[name]
        print(f"    {name:20s} {m['mae']:8.3f} {m['rmse']:8.3f} {m['r2']:8.3f} "
              f"{m['mae_nonempty']:8.3f} {m['within_1_veh']:9.1%}")
    blind = min(block[n]["mae"] for n in order[1:])
    gain = 1 - block["probe"]["mae"] / blind if blind else 0.0
    blind_nz = min(block[n]["mae_nonempty"] for n in order[1:])
    gain_nz = 1 - block["probe"]["mae_nonempty"] / blind_nz if blind_nz else 0.0
    # On a target that is ~87% zero, overall MAE is dominated by the zeros and
    # "predict zero" is hard to beat on it. The non-empty figure is the one that
    # says whether the model can actually read a queue.
    verdict = "reads the image" if gain_nz > 0.15 else "NO BETTER THAN BLIND"
    print(
        f"    -> vs best blind baseline: overall MAE {describe(gain)}, "
        f"non-empty-lane MAE {describe(gain_nz)}  [{verdict}]"
    )


def describe(gain: float) -> str:
    return f"{gain:.0%} lower" if gain >= 0 else f"{-gain:.0%} higher"


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Queue-length perception probe.")
    parser.add_argument("--dataset", default="beihuan_pilot", help="Dataset name under data/samples.")
    parser.add_argument("--target", default="queued",
                        choices=["queued", "vehicles", "queue_m", "occupancy"],
                        help="Per-lane quantity to regress. 'vehicles' is dense and separates a weak feature from a sparse target.")
    parser.add_argument("--loss", default="mse", choices=["mse", "l1", "huber"],
                        help="MSE by default: L1 on a sparse target collapses to predicting zero.")
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--depth", type=int, default=2, help="1 = linear probe.")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--test-fraction", type=float, default=0.25)
    parser.add_argument("--drop-saturated", action="store_true",
                        help="Exclude lanes whose queue reaches the image edge (label is a lower bound).")
    parser.add_argument("--junction", nargs="+", default=None, help="Restrict to these junctions.")
    parser.add_argument("--plan", nargs="+", default=None, help="Restrict to these signal plans.")
    parser.add_argument("--pooled-only", action="store_true",
                        help="Skip the per-junction breakdown; fit one shared head only.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    results = run(args)
    out = Path(args.out) if args.out else REPORTS_ROOT / f"probe_queue_{args.dataset}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\n[summary] -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
