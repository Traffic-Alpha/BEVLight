'''
@Author: WANG Maonan
@Date: 2026-08-28
@Description: Driving one episode: with the agent, with a baseline, for a score.

The four ways an episode gets run during training, kept together because they
have to stay comparable -- the agent is judged against `rollout_controller` on
the same junction, plan, demand and seed, and a difference in how the loop is
driven would show up as a difference in the method.
'''

from __future__ import annotations

from .replay import to_batch


def rollout_policy(agent, junction, plan, demand, seed, device, reward, steps=None,
                   observe="window") -> dict:
    """One greedy episode. The metrics are the ones the results tables report."""
    from ...env.gym_env import JunctionEnv

    env = JunctionEnv(junction, plan, demand, seed=seed, num_seconds=steps,
                      render=False, allow_any_scenario=True, reward=reward,
                      observe=observe)
    try:
        observation, _, done, _ = env.reset()
        while not done:
            action = int(agent.act(to_batch([observation], device), greedy=True)[0])
            observation, _, done, _ = env.step(action)
        return env.summary()
    finally:
        env.close()


def rollout_controller(spec, junction, plan, demand, seed, reward, steps=None) -> dict:
    """The same episode under a rule-based controller, for the paired comparison."""
    from ...env.gym_env import JunctionEnv
    from ...eval.compare import build_controller

    env = JunctionEnv(junction, plan, demand, seed=seed, num_seconds=steps,
                      render=False, allow_any_scenario=True, reward=reward)
    controller = build_controller(spec)
    controller.reset(env.signal_plan)
    try:
        _, _, done, _ = env.reset()
        while not done:
            action = env.signal_plan.phases.index(
                controller.act(env._pending, env.signal_plan)
            )
            _, _, done, _ = env.step(action)
        return env.summary()
    finally:
        env.close()


def preflight(junction, plan, demand, reward, steps=None) -> dict:
    """Two episodes, before spending hours: does the reward order what it should?

    Deliberately not the full `bevlight rl preflight`, which needs several
    controllers to separate "correct" from "correct by construction". This is the
    cheaper guard against the failure that wastes a whole run: a reward that is
    flat, inverted, or identical for a good and a bad controller. The first
    version of this project's reward telescoped to exactly zero for every
    controller, and a day of training with a flat curve is indistinguishable from
    a bad learning rate.
    """
    from ..preflight import rollout

    rows = {spec: rollout(junction, plan, demand, spec, 7, steps)
            for spec in ("fixed_time", "max_pressure")}
    good, bad = rows["max_pressure"]["reward_live"], rows["fixed_time"]["reward_live"]
    margin = (good - bad) / abs(bad) if bad else 0.0
    ordered = good > bad and margin > 0.05
    print(f"[reward] max_pressure {good:+.5f} vs fixed_time {bad:+.5f} "
          f"(margin {margin:+.1%}) -> {'ordered correctly' if ordered else 'NOT ORDERED'}")
    print(f"[reward] travel time  max_pressure {rows['max_pressure']['avg_travel_time_s']:.2f}s "
          f"vs fixed_time {rows['fixed_time']['avg_travel_time_s']:.2f}s")
    return {"ordered": ordered, "margin": margin,
            "max_pressure": good, "fixed_time": bad}


def evaluate(agent, junction, plan, demand, seeds, device, reward, baselines,
             steps=None, observe="window") -> dict:
    """The gate's actual question, paired seed by seed.

    Common random numbers: the policy and max-pressure are scored on the *same*
    traffic, and the difference is taken per seed before averaging. The demand
    realisation is the largest source of variance in these numbers and pairing
    removes it outright — without that, a 1-3% difference is invisible under a
    noise floor measured at 1.2 s.
    """
    rows = [rollout_policy(agent, junction, plan, demand, seed, device, reward,
                           steps, observe)
            for seed in seeds]

    def mean(values):
        return float(sum(values) / len(values)) if values else 0.0

    result = {"policy": {k: mean([r[k] for r in rows])
                         for k in ("avg_travel_time_s", "avg_waiting_time_s",
                                   "avg_queue_veh", "throughput", "switch_rate")}}
    for spec, reference in baselines.items():
        deltas = [r["avg_travel_time_s"] - b["avg_travel_time_s"]
                  for r, b in zip(rows, reference)]
        result[spec] = {
            "travel": mean([b["avg_travel_time_s"] for b in reference]),
            "paired_delta_travel": round(mean(deltas), 3),
            "per_seed_delta": [round(d, 3) for d in deltas],
            "wins": sum(1 for d in deltas if d < 0),
        }
    return result
