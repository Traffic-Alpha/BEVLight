'''
@Author: WANG Maonan
@Date: 2026-08-20
@Description: Per-lane traffic state, restricted to what the BEV window shows.

This is the one place where "what the camera sees" is turned into numbers, and
both consumers read it:

  * the expert, so that observation -> action stays well-defined. An expert that
    reacts to a queue extending past the image edge is asking the vision model to
    imitate something it cannot see.
  * the auxiliary regression labels, for the same reason in reverse: a label must
    not describe vehicles outside the input.

Sharing one extractor is what guarantees the two cannot drift apart.

A lane is measured from the junction outwards: for an incoming lane the origin is
the stop line at the lane's end, for an outgoing lane it is the lane's start.
Only the first `LaneMask.visible_length_m(lane)` metres count.
@LastEditTime: 2026-08-20
'''

from __future__ import annotations

from dataclasses import dataclass, field

# A vehicle counts as queued below this speed. SUMO's own halting threshold.
HALTING_SPEED_MS = 0.1


@dataclass(frozen=True)
class LaneState:
    """Traffic state of one lane, inside the BEV window only."""

    lane_id: str
    role: str
    visible_m: float
    vehicles: int = 0          # vehicles present in the visible stretch
    queued: int = 0            # of those, halting
    queue_m: float = 0.0       # furthest halting vehicle back from the junction
    occupancy: float = 0.0     # fraction of the visible stretch covered by vehicles

    @property
    def queue_saturated(self) -> bool:
        """The queue reaches the edge of the image, so its true length is unknown."""
        return self.visible_m > 0 and self.queue_m >= self.visible_m - 1.0


@dataclass
class JunctionObservation:
    """Everything a controller may look at, for one simulation second."""

    time: float
    lanes: dict[str, LaneState] = field(default_factory=dict)
    phase_index: int = 0
    time_in_phase: float = 0.0
    can_act: bool = False

    def lane(self, lane_id: str) -> LaneState:
        return self.lanes[lane_id]

    def queued(self, lane_ids) -> int:
        return sum(self.lanes[l].queued for l in lane_ids if l in self.lanes)

    def vehicles(self, lane_ids) -> int:
        return sum(self.lanes[l].vehicles for l in lane_ids if l in self.lanes)

    def occupancy(self, lane_ids) -> float:
        seen = [self.lanes[l].occupancy for l in lane_ids if l in self.lanes]
        return sum(seen) / len(seen) if seen else 0.0

    def saturated_lanes(self) -> list[str]:
        return [l for l, st in self.lanes.items() if st.queue_saturated]


class ObservationExtractor:
    """Turns raw tshub vehicle states into per-lane visible-window state."""

    def __init__(self, mask, tls_id: str, full_lane: bool = False):
        self.mask = mask
        self.tls_id = tls_id
        # `full_lane` measures the whole approach instead of the stretch the BEV
        # window shows. It is not a camera setting and never labels anything: it
        # exists so a control experiment can hold everything else fixed and vary
        # only how much of the junction the policy is allowed to see. Anything
        # the vision model will have to reproduce must come from the default.
        self.full_lane = bool(full_lane)
        self.visible = (
            {record["lane_id"]: float(record["length"]) for record in mask.lanes}
            if full_lane else mask.visible_length_m()
        )
        self.roles = {r["lane_id"]: r["role"] for r in mask.lanes}
        self.lengths = {r["lane_id"]: float(r["length"]) for r in mask.lanes}

    def distance_from_junction(self, lane_id: str, lane_position: float) -> float:
        """How far back from the junction a vehicle sits, in metres.

        Incoming lanes run *towards* the junction, so the stop line is at the far
        end of the lane and the distance is measured backwards from it. Outgoing
        lanes run away from it, so `lane_position` already is the distance.
        """
        if self.roles.get(lane_id) == "incoming":
            return self.lengths[lane_id] - float(lane_position)
        return float(lane_position)

    def __call__(self, states: dict) -> JunctionObservation:
        tls = states["tls"][self.tls_id]
        obs = JunctionObservation(
            time=float(states.get("sim_step", 0.0) or 0.0),
            phase_index=int(tls["this_phase_index"]),
            can_act=bool(tls["can_perform_action"]),
        )

        counts: dict[str, list] = {lane_id: [] for lane_id in self.visible}
        for vehicle in states["vehicle"].values():
            lane_id = vehicle["lane_id"]
            bucket = counts.get(lane_id)
            if bucket is None:
                continue
            distance = self.distance_from_junction(lane_id, vehicle["lane_position"])
            if distance > self.visible[lane_id]:
                continue          # physically there, but outside the image
            bucket.append((distance, float(vehicle["speed"]), float(vehicle["length"])))

        for lane_id, entries in counts.items():
            visible_m = self.visible[lane_id]
            halting = [e for e in entries if e[1] < HALTING_SPEED_MS]
            obs.lanes[lane_id] = LaneState(
                lane_id=lane_id,
                role=self.roles[lane_id],
                visible_m=visible_m,
                vehicles=len(entries),
                queued=len(halting),
                queue_m=round(max((e[0] for e in halting), default=0.0), 3),
                occupancy=round(
                    min(1.0, sum(e[2] for e in entries) / visible_m) if visible_m > 0 else 0.0,
                    4,
                ),
            )
        return obs
