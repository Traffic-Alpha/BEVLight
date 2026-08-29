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
from ._internal.replay import NStepAccumulator, ReplayBuffer, next_batch, to_batch
from ._internal.rollout import evaluate, preflight, rollout_controller


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
