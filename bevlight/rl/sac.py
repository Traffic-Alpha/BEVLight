'''
@Author: WANG Maonan
@Date: 2026-08-28
@Description: Discrete SAC for the privileged teacher.

This is the gate, not the method. The question it answers is narrow and has to be
asked before anything expensive is built on top: *on this junction, under this
demand, can a learned controller beat max-pressure at all* — and if it can, by how
much. A negative answer here makes distillation pointless, and it costs hours to
find out rather than the weeks the vision loop would have taken.

Three choices are what make it cheap enough to be a gate.

**No renderer.** The environment is `JunctionEnv(render=False)`, the same world
`run_episode` drives and `tests/test_gym_env.py` pins metric-for-metric. A
1000-second episode costs 3-14 s of SUMO against 22 minutes per update through
Panda, and the profile says 68% of even that is TraCI, so there is nothing left
to optimise in our own code.

**Off-policy, with a replay buffer.** PPO uses each sample for a few epochs and
discards it, which is the wrong trade at any sample cost above free — and it was
the specific complaint the previous RL attempt ended on. Every transition here is
replayed until it falls out of the buffer.

**Q per candidate, not a single value.** The action space is 3 or 4 phases, so a
critic can score all of them in one forward. That is what makes the teacher
distillable: the student can be trained on the *margin* between phases, which says
which decisions matter, where an argmax label says only what was chosen.

The observation is restricted to what the BEV window shows. The reward is not:
see `JunctionEnv.reward_kind`. A reward is a training-time signal no deployed
policy computes, and paying the teacher on a queue that stops growing at the image
edge would teach it that a jam is cheaper once it is out of frame.
@LastEditTime: 2026-08-28
'''

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ..model.teacher import TeacherNet, teacher_config

# Everything about a junction that does not change during an episode. Stored once
# per (junction, plan) and referenced by index: at ~5 KB a transition it would
# otherwise be four fifths of the replay buffer, describing wiring that is the
# same in every row.
STRUCTURE_KEYS = (
    "lane_valid", "movement_in_index", "movement_in_weight",
    "movement_out_index", "movement_out_weight", "movement_valid",
    "phase_members", "phase_member_valid", "phase_valid",
)


@dataclass
class SACConfig:
    """Post-hoc note on every number here: none of them are tuned. This is a gate."""

    total_steps: int = 60_000            # environment decisions, summed over workers
    num_envs: int = 4
    buffer_size: int = 60_000
    batch_size: int = 256
    start_steps: int = 2_000             # uniform random actions, to fill the buffer
    # Gradient steps per *transition* collected — the update-to-data ratio, and
    # the only knob that changes this run's wall clock. A measured update costs
    # 79 ms compiled and does not get cheaper with a smaller model, so at the
    # usual ratio of 1 the throughput is ~13 transitions/s no matter how many
    # workers collect them: adding processes past the point where simulation
    # hides under the gradient step buys nothing. Spending the other 76 cores
    # means lowering this, trading gradient steps for a wider, fresher buffer.
    updates_per_step: float = 1.0
    # 0.95, not 0.99. What drives the policy is the *difference* in Q between
    # candidate phases, and that difference is order 0.1 while |Q| under 0.99 runs
    # to 20-30: the critic would need better than one percent relative accuracy
    # before the actor sees a signal at all. 0.95 is an effective horizon of 20
    # decisions — 200 s, ample for a signal plan — against an episode of 66, so
    # nothing real is being discarded, only the noise past it.
    gamma: float = 0.95
    # Decisions are 10 s apart and queue dynamics are slow: releasing a phase too
    # early shows up three or four decisions later. One-step TD has to walk that
    # credit back one bootstrap at a time, which is what a flat return curve on a
    # converging critic looks like.
    n_step: int = 3
    tau: float = 0.005
    lr: float = 3e-4
    alpha_lr: float = 3e-4
    # Discrete SAC's entropy target as a fraction of log K. K is 3 or 4 here, so
    # it is set per sample from that sample's candidate count rather than fixed.
    target_entropy_ratio: float = 0.25
    grad_clip: float = 1.0
    model_dim: int = 128
    embed_dim: int = 64
    eval_every: int = 5_000
    eval_seeds: tuple = (101, 102, 103)
    checkpoint_every: int = 10_000
    reward: str = "visible_queue"
    # What the policy is allowed to read. `window` is the deployable setting;
    # `full_lane` is the control arm that separates "the learner cannot do better"
    # from "the BEV window does not carry enough". Only the policy's input moves —
    # the reward and every baseline stay where they are.
    observe: str = "window"
    compile: bool = True
    seed: int = 7


class NStepAccumulator:
    """Turns one environment's single steps into n-step transitions.

    One per environment, because an n-step window must never span two episodes —
    and with auto-reset the boundary is invisible in the observation stream. On
    an episode end the whole queue is flushed, each remaining start with its own
    shorter horizon, so the last few decisions of an episode are not discarded.

    The bootstrap discount travels with the transition rather than being assumed:
    at an episode end a window may be shorter than n, and using gamma^n there
    would discount a value that is only m steps away.
    """

    def __init__(self, n: int, gamma: float):
        from collections import deque

        self.n, self.gamma = int(n), float(gamma)
        self.pending: deque = deque()

    def push(self, observation, action, reward, next_observation, terminal, done):
        """-> the n-step transitions this step completed, possibly none."""
        self.pending.append((observation, action, float(reward)))
        if done:
            emitted = self._drain(len(self.pending), next_observation, terminal)
            self.pending.clear()
            return emitted
        if len(self.pending) >= self.n:
            emitted = self._drain(1, next_observation, False)
            self.pending.popleft()
            return emitted
        return []

    def _drain(self, count: int, final_observation, terminal: bool) -> list:
        entries = list(self.pending)
        out = []
        for start in range(count):
            total, discount = 0.0, 1.0
            for _, _, reward in entries[start:]:
                total += discount * reward
                discount *= self.gamma
            observation, action, _ = entries[start]
            # `discount` is now gamma ** (steps in this window) — the factor the
            # bootstrapped value must be multiplied by.
            out.append((observation, action, total, final_observation,
                        terminal, discount))
        return out


class ReplayBuffer:
    """Transitions, with the junction wiring factored out.

    `lane_state` is kept in float16: it holds queue counts, an occupancy fraction
    and a flag, none of which carry seven significant digits, and it is otherwise
    four fifths of the memory.
    """

    def __init__(self, capacity: int, window: int, max_lanes: int, lane_dim: int):
        self.capacity = int(capacity)
        self.size, self.cursor = 0, 0
        shape = (self.capacity, window, max_lanes, lane_dim)
        self.lane_state = np.zeros(shape, dtype=np.float16)
        self.next_lane_state = np.zeros(shape, dtype=np.float16)
        self.current_phase = np.zeros(self.capacity, dtype=np.int64)
        self.next_current_phase = np.zeros(self.capacity, dtype=np.int64)
        self.time_in_phase = np.zeros(self.capacity, dtype=np.float32)
        self.next_time_in_phase = np.zeros(self.capacity, dtype=np.float32)
        self.action = np.zeros(self.capacity, dtype=np.int64)
        self.reward = np.zeros(self.capacity, dtype=np.float32)
        # True only when the network really drained. A horizon cut leaves traffic
        # on the road, and its value has to be bootstrapped rather than zeroed.
        self.terminal = np.zeros(self.capacity, dtype=np.float32)
        # gamma ** (steps in this transition's window). Not a constant: a window
        # cut short by the end of an episode bootstraps from closer in.
        self.discount = np.zeros(self.capacity, dtype=np.float32)
        self.structure_id = np.zeros(self.capacity, dtype=np.int64)
        self._structure_index: dict = {}
        self._structures: list = []
        self._stacked: dict | None = None

    def structure_slot(self, key, observation: dict) -> int:
        if key not in self._structure_index:
            self._structure_index[key] = len(self._structures)
            self._structures.append({k: observation[k] for k in STRUCTURE_KEYS})
            self._stacked = None
        return self._structure_index[key]

    def stack_structures(self) -> dict:
        """Cached: rebuilt only when a junction the buffer has not seen arrives."""
        if self._stacked is None:
            self._stacked = {k: np.stack([s[k] for s in self._structures])
                             for k in STRUCTURE_KEYS}
        return self._stacked

    def add(self, key, observation, action, reward, next_observation, terminal,
            discount) -> None:
        i = self.cursor
        self.lane_state[i] = observation["lane_state"]
        self.next_lane_state[i] = next_observation["lane_state"]
        self.current_phase[i] = observation["current_phase"]
        self.next_current_phase[i] = next_observation["current_phase"]
        self.time_in_phase[i] = observation["time_in_phase"]
        self.next_time_in_phase[i] = next_observation["time_in_phase"]
        self.action[i] = action
        self.reward[i] = reward
        self.terminal[i] = float(terminal)
        self.discount[i] = float(discount)
        self.structure_id[i] = self.structure_slot(key, observation)
        self.cursor = (self.cursor + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device) -> dict:
        index = np.random.randint(0, self.size, size=batch_size)
        structures = self.stack_structures()
        ids = self.structure_id[index]

        def tensor(array, dtype=torch.float32):
            return torch.as_tensor(array, dtype=dtype, device=device)

        batch = {k: tensor(structures[k][ids],
                           torch.int64 if structures[k].dtype == np.int64 else torch.float32)
                 for k in STRUCTURE_KEYS}
        batch.update(
            lane_state=tensor(self.lane_state[index].astype(np.float32)),
            next_lane_state=tensor(self.next_lane_state[index].astype(np.float32)),
            current_phase=tensor(self.current_phase[index], torch.int64),
            next_current_phase=tensor(self.next_current_phase[index], torch.int64),
            time_in_phase=tensor(self.time_in_phase[index]),
            next_time_in_phase=tensor(self.next_time_in_phase[index]),
            action=tensor(self.action[index], torch.int64),
            reward=tensor(self.reward[index]),
            terminal=tensor(self.terminal[index]),
            discount=tensor(self.discount[index]),
        )
        return batch


def to_batch(observations: list[dict], device) -> dict:
    """Stack environment observations into what `TeacherNet` consumes."""
    batch = {}
    for key in STRUCTURE_KEYS:
        stacked = np.stack([o[key] for o in observations])
        dtype = torch.int64 if stacked.dtype == np.int64 else torch.float32
        batch[key] = torch.as_tensor(stacked, dtype=dtype, device=device)
    batch["lane_state"] = torch.as_tensor(
        np.stack([o["lane_state"] for o in observations]), dtype=torch.float32, device=device
    )
    batch["current_phase"] = torch.as_tensor(
        [o["current_phase"] for o in observations], dtype=torch.int64, device=device
    )
    batch["time_in_phase"] = torch.as_tensor(
        [o["time_in_phase"] for o in observations], dtype=torch.float32, device=device
    )
    return batch


def next_batch(batch: dict) -> dict:
    """The same wiring, the next state. The junction did not change mid-episode."""
    return {**batch,
            "lane_state": batch["next_lane_state"],
            "current_phase": batch["next_current_phase"],
            "time_in_phase": batch["next_time_in_phase"]}


def policy(scores: torch.Tensor, phase_valid: torch.Tensor):
    """Scores -> `(probabilities, log probabilities)` over the real candidates only.

    Padded candidates arrive at -1e9 from the decision layer, so they take no
    softmax mass; they are zeroed here as well so that a sum over K never
    multiplies a zero probability by a large negative log and calls it a gradient.
    """
    valid = phase_valid.bool()
    log_probabilities = torch.log_softmax(scores.masked_fill(~valid, -1e9), dim=-1)
    # Exponentiate first: zeroing the logs before this would turn every padded
    # candidate into exp(0) = 1 and hand the padding all the probability mass.
    probabilities = log_probabilities.exp() * valid.float()
    log_probabilities = log_probabilities.masked_fill(~valid, 0.0)
    return probabilities, log_probabilities


class DiscreteSAC:
    """Soft actor-critic over a small, masked, variable-size action set.

    Discrete rather than the usual continuous SAC because a phase is a choice
    among three or four candidates, and that has a consequence worth naming: the
    critic scores *every* candidate in one forward, so the expectations SAC needs
    are exact sums rather than samples. There is no reparameterisation trick here
    and no sampling noise in the actor's gradient.
    """

    def __init__(self, config: SACConfig, device):
        import copy

        cfg = teacher_config(config.model_dim, config.embed_dim)
        self.config, self.device = config, device
        self.actor = TeacherNet(cfg).to(device)
        self.q1 = TeacherNet(cfg).to(device)
        self.q2 = TeacherNet(cfg).to(device)
        self.q1_target = copy.deepcopy(self.q1).requires_grad_(False)
        self.q2_target = copy.deepcopy(self.q2).requires_grad_(False)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=config.lr)
        self.critic_opt = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=config.lr
        )
        # Tuned, not fixed: a behaviour-cloned start collapses to near-zero
        # entropy and a hand-set temperature is the wrong knob to discover that
        # with. Starting at alpha = 1 rather than lower because this one starts
        # from scratch and has nothing to preserve.
        self.log_alpha = torch.zeros(1, device=device, requires_grad=True)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=config.alpha_lr)

        # Measured: 128 ms an update at model_dim 128 and 126 ms at 64. A model
        # whose cost does not move with its size is not doing arithmetic, it is
        # launching kernels — the same finding `train/run.py` records for the
        # behaviour-cloning step. Compiling is the lever that applies.
        if config.compile:
            for name in ("actor", "q1", "q2", "q1_target", "q2_target"):
                setattr(self, name, torch.compile(getattr(self, name)))

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    @torch.no_grad()
    def act(self, batch: dict, greedy: bool = False) -> np.ndarray:
        probabilities, _ = policy(self.actor(batch), batch["phase_valid"])
        if greedy:
            return probabilities.argmax(dim=-1).cpu().numpy()
        return torch.multinomial(probabilities, 1).squeeze(-1).cpu().numpy()

    @torch.no_grad()
    def _soft_update(self) -> None:
        """Polyak averaging, fused.

        A Python loop over the parameters issues two kernels per tensor — about
        four hundred launches per update for two critics — against a model whose
        arithmetic is a few milliseconds. `_foreach_` does the same work in two.
        """
        tau = self.config.tau
        for online, target in ((self.q1, self.q1_target), (self.q2, self.q2_target)):
            source = list(online.parameters())
            destination = list(target.parameters())
            torch._foreach_mul_(destination, 1 - tau)
            torch._foreach_add_(destination, source, alpha=tau)

    def update(self, batch: dict) -> dict:
        valid = batch["phase_valid"]
        alpha = self.alpha.detach()

        with torch.no_grad():
            following = next_batch(batch)
            probabilities, log_probabilities = policy(
                self.actor(following), valid
            )
            q_target = torch.min(self.q1_target(following), self.q2_target(following))
            # An exact expectation over candidates, not a sample of one.
            value = (probabilities * (q_target - alpha * log_probabilities)).sum(-1)
            # `terminal` is set only when the network actually drained. A horizon
            # cut still has traffic on it and must bootstrap, or every long
            # episode teaches the critic that the world ends worthless.
            # `reward` is the discounted sum over this transition's window and
            # `discount` is gamma to the length of that window, so this is the
            # n-step target with no assumption that the window was full.
            target = batch["reward"] + batch["discount"] * (1 - batch["terminal"]) * value

        action = batch["action"].unsqueeze(-1)
        scores1, scores2 = self.q1(batch), self.q2(batch)
        q1 = scores1.gather(1, action).squeeze(-1)
        q2 = scores2.gather(1, action).squeeze(-1)
        critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)

        # The actor is scored by the critic that just produced this loss, rather
        # than by a second forward through the critic after its step. Two of the
        # eight network passes an update makes, for a one-gradient-step-stale Q —
        # and the target networks the actor ultimately chases move at tau=0.005,
        # which is three orders of magnitude slower than that staleness.
        q_for_actor = torch.min(scores1, scores2).detach()

        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        critic_grad = torch.nn.utils.clip_grad_norm_(
            list(self.q1.parameters()) + list(self.q2.parameters()), self.config.grad_clip
        )
        self.critic_opt.step()

        probabilities, log_probabilities = policy(self.actor(batch), valid)
        actor_loss = (
            probabilities * (alpha * log_probabilities - q_for_actor)
        ).sum(-1).mean()

        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        actor_grad = torch.nn.utils.clip_grad_norm_(
            self.actor.parameters(), self.config.grad_clip
        )
        self.actor_opt.step()

        entropy = -(probabilities * log_probabilities).sum(-1)
        # K varies by junction and by plan, so the entropy target is read off each
        # sample's own candidate count rather than assumed constant.
        candidates = valid.sum(-1).clamp(min=1)
        target_entropy = self.config.target_entropy_ratio * candidates.log()
        alpha_loss = (self.alpha * (entropy - target_entropy).detach()).mean()
        self.alpha_opt.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_opt.step()

        self._soft_update()
        return {
            "critic_loss": critic_loss.item(), "actor_loss": actor_loss.item(),
            "alpha": self.alpha.detach().item(), "entropy": entropy.mean().item(),
            "q_mean": q1.mean().item(), "td_error": (q1 - target).abs().mean().item(),
            "critic_grad": critic_grad.item(), "actor_grad": actor_grad.item(),
        }

    def state_dict(self) -> dict:
        return {"actor": self.actor.state_dict(), "q1": self.q1.state_dict(),
                "q2": self.q2.state_dict(), "log_alpha": self.log_alpha.detach().cpu(),
                "config": asdict(self.config)}


def rollout_policy(agent, junction, plan, demand, seed, device, reward, steps=None,
                   observe="window") -> dict:
    """One greedy episode. The metrics are the ones the results tables report."""
    from ..env.gym_env import JunctionEnv

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
    from ..env.gym_env import JunctionEnv
    from ..eval.compare import build_controller

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
    from .preflight import rollout

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


def train(config: SACConfig, junction: str, plan: str, demand: str,
          run_dir: Path | None = None, device=None, steps: int | None = None,
          run_preflight: bool = True) -> dict:
    """Train the teacher, and score it against max-pressure as it goes."""
    from ..env.vector import make_envs

    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    if run_preflight:
        check = preflight(junction, plan, demand, config.reward, steps)
        if not check["ordered"]:
            raise SystemExit(
                "The reward does not prefer max-pressure to fixed-time on this "
                "scenario by a clear margin. Training against it would optimise "
                "something that is not control quality."
            )

    print(f"[baseline] scoring the rule-based controllers on seeds {list(config.eval_seeds)}")
    baselines = {
        spec: [rollout_controller(spec, junction, plan, demand, seed, config.reward, steps)
               for seed in config.eval_seeds]
        for spec in ("max_pressure", "fixed_time")
    }
    for spec, rows in baselines.items():
        travel = sum(r["avg_travel_time_s"] for r in rows) / len(rows)
        print(f"[baseline] {spec:14s} travel={travel:7.2f}s")

    envs = make_envs(config.num_envs, render=False, junction=junction, plan=plan,
                     demand=demand, reward=config.reward, observe=config.observe,
                     num_seconds=steps, idle_timeout=30, seed=config.seed)
    agent = DiscreteSAC(config, device)
    observations = [env.reset()[0] for env in envs]
    buffer = ReplayBuffer(config.buffer_size, window=observations[0]["lane_state"].shape[0],
                          max_lanes=observations[0]["lane_state"].shape[1],
                          lane_dim=observations[0]["lane_state"].shape[2])

    history, episode_returns, returns = [], [0.0] * len(envs), []
    update_debt = 0.0
    # One per environment: an n-step window that spanned two episodes would carry
    # reward across a reset, and with auto-reset that boundary is invisible in the
    # observation stream.
    accumulators = [NStepAccumulator(config.n_step, config.gamma) for _ in envs]
    started = time.time()
    collected, stats = 0, {}
    next_eval, next_checkpoint = config.eval_every, config.checkpoint_every

    while collected < config.total_steps:
        if collected < config.start_steps:
            actions = [int(np.random.randint(int(o["phase_valid"].sum())))
                       for o in observations]
        else:
            actions = agent.act(to_batch(observations, device)).tolist()

        # Issue every action before training on anything. The simulators are in
        # their own processes, so the gradient steps below run *while* they
        # advance rather than after: with N workers the two costs overlap instead
        # of adding, and the wall clock becomes max(simulate, learn) rather than
        # their sum. This is the whole reason the vector environment exists here —
        # collecting in parallel and then idling every worker through the update
        # would leave most of the machine doing nothing most of the time.
        for env, action in zip(envs, actions):
            env.step_async(int(action))

        if buffer.size >= config.batch_size and collected >= config.start_steps:
            # Fractional ratios accumulate rather than round to zero, so 0.25
            # means one update every fourth vector step and not none at all.
            update_debt += config.updates_per_step * len(envs)
            while update_debt >= 1.0:
                stats = agent.update(buffer.sample(config.batch_size, device))
                update_debt -= 1.0

        for i, env in enumerate(envs):
            following, reward, done, info = env.step_wait()
            # On `done` the environment has already restarted, so the state this
            # transition ends in travels in `info` — see `env/vector.py`.
            successor = info["terminal_observation"] if done else following
            for entry in accumulators[i].push(
                observations[i], actions[i], reward, successor,
                bool(done and info.get("drained")), done,
            ):
                buffer.add((junction, plan), *entry)
            # The undiscounted single-step return, so the logged number stays
            # comparable with the reward preflight and with the previous run.
            episode_returns[i] += reward
            if done:
                returns.append(episode_returns[i])
                episode_returns[i] = 0.0
            observations[i] = following
            collected += 1

        if collected >= next_eval and stats:
            next_eval += config.eval_every
            scores = evaluate(agent, junction, plan, demand, config.eval_seeds,
                              device, config.reward, baselines, steps, config.observe)
            entry = {"steps": collected, "episodes": len(returns),
                     "steps_per_s": round(collected / max(1e-9, time.time() - started), 1),
                     "return": round(float(np.mean(returns[-10:])), 4) if returns else None,
                     "elapsed_s": round(time.time() - started, 1),
                     **{k: round(v, 4) for k, v in stats.items()}, "eval": scores}
            history.append(entry)
            print(
                f"step {collected:6d}  R={entry['return']}  H={stats['entropy']:.3f} "
                f"alpha={stats['alpha']:.3f} q={stats['q_mean']:+.3f} "
                f"td={stats['td_error']:.3f} {entry['steps_per_s']:.0f}/s  |  travel={scores['policy']['avg_travel_time_s']:.2f}s "
                f"vs MP {scores['max_pressure']['travel']:.2f}s "
                f"(paired {scores['max_pressure']['paired_delta_travel']:+.2f}s, "
                f"{scores['max_pressure']['wins']}/{len(config.eval_seeds)} seeds better)  "
                f"sw={scores['policy']['switch_rate']:.2f}  {entry['elapsed_s']:.0f}s",
                flush=True,
            )
            if run_dir:
                (run_dir / "history.json").write_text(json.dumps(history, indent=2))

        if run_dir and collected >= next_checkpoint:
            next_checkpoint += config.checkpoint_every
            torch.save(agent.state_dict(), run_dir / f"teacher_{collected:06d}.pt")

    for env in envs:
        env.close()
    if run_dir:
        torch.save(agent.state_dict(), run_dir / "teacher_final.pt")
        (run_dir / "history.json").write_text(json.dumps(history, indent=2))
    return {"history": history, "elapsed_s": round(time.time() - started, 1)}
