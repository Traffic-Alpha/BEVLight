'''
@Author: WANG Maonan
@Date: 2026-08-29
@Description: The per-second costs a reward can be built from, in one registry.

A reward is a hypothesis -- *optimising this produces good control* -- and the
way to test one cheaply is to measure several over the same rollout
(`bevlight.rl.preflight`). That only works if the candidate the probe measures
and the reward the learner is paid are the same arithmetic. They used to be two
implementations that a comment required to agree; this is the one they now
share, so "the probe measures something the learner never sees" stops being a
bug that can be written.

Every function here is a **cost for one simulated second**: higher is worse.
Both callers accumulate it over a decision interval and return
`-total / seconds` -- the integral, not the endpoint difference, because a
difference telescopes away over an episode that starts and ends empty and says
nothing about how long the queues persisted.

Costs are per incoming lane, so a 28-lane junction and a 12-lane one produce
gradients of the same scale rather than the larger junction dominating.

## What a reward may read, and why it is not what the policy may read

The observation is restricted to the BEV window because that is what a deployed
policy will have. The reward is not, because no deployed policy ever computes
one. The distinction is load-bearing where the visible queue saturates: at
Beijing_Beihuan under high density a random controller's visible queue
understates its real one by 43%, so a policy paid on the visible queue is paid
less for a jam that has grown past the image edge than for one still inside it.
`full_queue` and `full_wait` are legitimate rewards for exactly that reason, and
illegitimate as observations for the same one.
@LastEditTime: 2026-08-29
'''

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RewardContext:
    """Everything any cost below needs, gathered once per simulated second.

    `visible` is the window-clipped per-lane state -- the same
    `ObservationExtractor` output the policy and the labels use -- while
    `states` is the raw simulator truth a full-lane cost needs.
    """

    visible: object          # JunctionObservation over the BEV window
    states: dict             # raw tshub states
    incoming_lanes: object   # lane ids of the incoming approaches
    current_phase: str
    first_phase: str

    @property
    def visible_incoming(self) -> list:
        return [s for s in self.visible.lanes.values() if s.role == "incoming"]

    @property
    def lane_count(self) -> int:
        """Incoming lanes, from the mask rather than from the traffic.

        Never zero, and never dependent on what happens to be on the road: a
        divisor that moved with the traffic would make the cost of an empty
        approach depend on whether another approach was busy.
        """
        return max(1, len(self.visible_incoming))


def visible_queue(ctx: RewardContext) -> float:
    """Queued vehicles per incoming lane inside the BEV window, right now."""
    return sum(s.queued for s in ctx.visible_incoming) / ctx.lane_count


def visible_occupancy(ctx: RewardContext) -> float:
    """The same window, as a fraction of it covered rather than a count."""
    return sum(s.occupancy for s in ctx.visible_incoming) / ctx.lane_count


def full_queue(ctx: RewardContext) -> float:
    """Stopped vehicles per incoming lane over the whole lane.

    What the image cannot show, and the reason this is here: a signal that does
    not saturate is what stops a jam past the image edge from looking cheaper
    than one still inside it.
    """
    return sum(
        1 for v in ctx.states["vehicle"].values()
        if v["lane_id"] in ctx.incoming_lanes and float(v.get("speed", 0.0)) < 0.1
    ) / ctx.lane_count


def full_wait(ctx: RewardContext) -> float:
    """Accumulated waiting per incoming lane.

    Closer in shape to travel time than a queue count, and unlike a queue it
    keeps counting a vehicle that has been stopped for a long time.
    """
    return sum(
        float(v.get("waiting_time", 0.0)) for v in ctx.states["vehicle"].values()
        if v["lane_id"] in ctx.incoming_lanes
    ) / ctx.lane_count


def probe_constant_phase(ctx: RewardContext) -> float:
    """1 for every second not spent on the first phase. A known optimum.

    A diagnostic, never a control objective: it pays for holding one phase and
    ignores the traffic entirely, so its optimal policy is known in advance --
    always choose action 0. A learner that cannot find that in a few thousand
    decisions is broken, and one that finds it quickly has been cleared of the
    "the code is wrong" explanation for failing to beat max-pressure. Training a
    teacher on it produces a controller that stops the traffic on three
    approaches.
    """
    return 0.0 if ctx.current_phase == ctx.first_phase else 1.0


#: Name -> per-second cost. The name is what `--reward` takes and what a run's
#: config records, so it is part of the experiment record and does not change.
REWARDS = {
    "visible_queue": visible_queue,
    "visible_occupancy": visible_occupancy,
    "full_queue": full_queue,
    "full_wait": full_wait,
    "probe_constant_phase": probe_constant_phase,
}

#: The candidates a reward preflight ranks. `probe_constant_phase` is excluded
#: on purpose: it is not a hypothesis about control, so ranking it against
#: travel time would be measuring nothing.
CANDIDATES = ("visible_queue", "full_queue", "full_wait", "visible_occupancy")
