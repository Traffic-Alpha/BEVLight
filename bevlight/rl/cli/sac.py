'''
@Author: WANG Maonan
@Date: 2026-08-29
@Description: Discrete SAC for the privileged teacher.

The command over `rl.sac`. Every knob lands in a `SACConfig`, which the run
directory records, so a finished run says what produced it.
'''

from __future__ import annotations

import json
from dataclasses import asdict

from ..sac import SACConfig, train


def parse_args(argv=None):
    import argparse

    defaults = SACConfig()
    parser = argparse.ArgumentParser(
        description="Train a privileged structured-state teacher, and score it "
                    "against max-pressure as it goes.")
    parser.add_argument("--junction", default="Beijing_Pinganli")
    parser.add_argument("--plan", default="easy")
    parser.add_argument("--demand", default="high_density")
    parser.add_argument("--run", default=None, help="Run name under runs/train/.")
    parser.add_argument("--total-steps", type=int, default=defaults.total_steps)
    parser.add_argument("--num-envs", type=int, default=defaults.num_envs)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--buffer-size", type=int, default=defaults.buffer_size)
    parser.add_argument("--start-steps", type=int, default=defaults.start_steps)
    parser.add_argument("--updates-per-step", type=float, default=defaults.updates_per_step,
                        help="Gradient steps per transition. 1.0 is the usual ratio and "
                             "caps throughput at ~13 transitions/s; lower it to use more workers.")
    parser.add_argument("--lr", type=float, default=defaults.lr)
    parser.add_argument("--gamma", type=float, default=defaults.gamma)
    parser.add_argument("--n-step", type=int, default=defaults.n_step,
                        help="Steps of reward before bootstrapping. 1 is plain TD.")
    parser.add_argument("--model-dim", type=int, default=defaults.model_dim)
    parser.add_argument("--target-entropy-ratio", type=float,
                        default=defaults.target_entropy_ratio)
    parser.add_argument("--observe", default=defaults.observe,
                        choices=["window", "full_lane"],
                        help="What the policy reads. full_lane is the control arm: it "
                             "widens the observation only, leaving the reward and every "
                             "baseline exactly where they are.")
    parser.add_argument("--reward", default=defaults.reward,
                        choices=["visible_queue", "full_queue", "probe_constant_phase"],
                        help="What the reward counts. The observation stays inside "
                             "the BEV window either way.")
    parser.add_argument("--eval-every", type=int, default=defaults.eval_every)
    parser.add_argument("--checkpoint-every", type=int, default=defaults.checkpoint_every)
    parser.add_argument("--eval-seed", nargs="+", type=int, default=list(defaults.eval_seeds))
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--steps", type=int, default=None,
                        help="Episode length. Default: the scenario's num_seconds.")
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-compile", action="store_true",
                        help="Skip torch.compile. The update is launch-bound, so this costs ~1.5x.")
    parser.add_argument("--no-preflight", action="store_true",
                        help="Skip the reward check. Only when it has just been run.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    from ...paths import TRAIN_RUNS_ROOT

    args = parse_args(argv)
    config = SACConfig(
        total_steps=args.total_steps, num_envs=args.num_envs,
        buffer_size=args.buffer_size, batch_size=args.batch_size,
        start_steps=args.start_steps, updates_per_step=args.updates_per_step,
        gamma=args.gamma, n_step=args.n_step, lr=args.lr, model_dim=args.model_dim,
        target_entropy_ratio=args.target_entropy_ratio, reward=args.reward,
        observe=args.observe,
        eval_every=args.eval_every, eval_seeds=tuple(args.eval_seed), seed=args.seed,
        checkpoint_every=args.checkpoint_every, compile=not args.no_compile,
    )
    run = args.run or f"teacher_{args.junction}_{args.plan}_{args.demand}"
    run_dir = TRAIN_RUNS_ROOT / run

    print(f"[plan] {args.junction}/{args.plan}_{args.demand}  reward={config.reward}  "
          f"steps={config.total_steps}  envs={config.num_envs}  render=off")
    print(f"[plan] observe={config.observe}  "
          f"(the reward and the baselines stay on the BEV window either way)")
    print(f"[plan] gamma={config.gamma} n_step={config.n_step} "
          f"entropy_target={config.target_entropy_ratio}xlogK utd={config.updates_per_step}")
    print(f"[plan] train seeds {config.seed}..{config.seed + config.num_envs - 1}, "
          f"eval seeds {list(config.eval_seeds)} (disjoint, and paired against the baselines)")
    print(f"[plan] -> {run_dir}")
    if args.dry_run:
        return 0

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(
        {**asdict(config), "junction": args.junction, "plan": args.plan,
         "demand": args.demand, "episode_steps": args.steps}, indent=2, default=list))
    result = train(config, args.junction, args.plan, args.demand, run_dir=run_dir,
                   device=args.device, steps=args.steps,
                   run_preflight=not args.no_preflight)
    print(f"\n[summary] {result['elapsed_s']}s -> {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
