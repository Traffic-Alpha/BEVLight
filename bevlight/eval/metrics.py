'''
@Author: WANG Maonan
@Date: 2026-08-20
@Description: Closed-loop control metrics for one episode.

These are what a controller is judged on — not action accuracy, which is
inflated by how often the expert keeps the current phase and in any case only
measures agreement with the expert rather than control quality.

Travel time is measured by watching vehicles appear and disappear rather than by
asking SUMO for arrivals, so the same accounting works whether the episode is
driven through tshub, a renderer, or a replay.

Vehicles still in the network when the episode ends are counted separately.
Ignoring them would flatter a controller that simply refuses to let traffic in:
its completed trips would all be fast ones.

Queues are recorded twice, and the distinction matters. The *visible* queue is
what the BEV window shows, and it is the right quantity for anything the model
consumes. It is the wrong quantity for judging control: it saturates at the edge
of the image, so a controller that lets queues grow past it looks artificially
good. Fixed-time at Beijing_Pinganli sits past the window ~48% of the time, so
its visible queue is capped while its real one keeps growing. Control tables use
the *full* queue.
@LastEditTime: 2026-08-20
'''

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EpisodeMetrics:
    """Accumulates control quality over one episode, one simulation second at a time."""

    # per-vehicle bookkeeping
    first_seen: dict[str, float] = field(default_factory=dict)
    last_seen: dict[str, float] = field(default_factory=dict)
    last_wait: dict[str, float] = field(default_factory=dict)
    completed: list[tuple[float, float]] = field(default_factory=list)  # (travel, wait)

    # per-second series
    queue_series: list[int] = field(default_factory=list)        # visible window only
    full_queue_series: list[int] = field(default_factory=list)   # whole lane
    incoming_lanes: set = field(default_factory=set)
    wait_series: list[float] = field(default_factory=list)
    vehicle_series: list[int] = field(default_factory=list)

    # signal behaviour
    switches: int = 0
    decisions: int = 0
    phase_counts: dict[int, int] = field(default_factory=dict)
    saturated_lane_seconds: int = 0

    _live: set = field(default_factory=set)

    def update(self, time: float, states: dict, obs=None) -> None:
        """Record one simulation second."""
        vehicles = states["vehicle"]
        present = set(vehicles)

        for vehicle_id, vehicle in vehicles.items():
            self.first_seen.setdefault(vehicle_id, time)
            self.last_seen[vehicle_id] = time
            self.last_wait[vehicle_id] = float(vehicle.get("accumulated_waiting_time", 0.0))

        for vehicle_id in self._live - present:          # left the network
            travel = self.last_seen[vehicle_id] - self.first_seen[vehicle_id]
            self.completed.append((travel, self.last_wait.get(vehicle_id, 0.0)))
        self._live = present

        self.vehicle_series.append(len(present))
        if self.incoming_lanes:
            self.full_queue_series.append(
                sum(
                    1
                    for vehicle in vehicles.values()
                    if vehicle["lane_id"] in self.incoming_lanes
                    and float(vehicle.get("speed", 0.0)) < 0.1
                )
            )
        if obs is not None:
            incoming = [st for st in obs.lanes.values() if st.role == "incoming"]
            self.queue_series.append(sum(st.queued for st in incoming))
            self.saturated_lane_seconds += sum(1 for st in incoming if st.queue_saturated)
        self.wait_series.append(
            sum(float(v.get("waiting_time", 0.0)) for v in vehicles.values())
        )

    def record_decision(self, previous_phase: int, action: int) -> None:
        self.decisions += 1
        self.phase_counts[action] = self.phase_counts.get(action, 0) + 1
        if action != previous_phase:
            self.switches += 1

    def finalize(self, time: float) -> None:
        """Close out vehicles still driving when the episode ended."""
        self.unfinished = len(self._live)
        self.unfinished_time = sum(time - self.first_seen[v] for v in self._live)

    def summary(self) -> dict:
        def mean(values):
            return float(sum(values) / len(values)) if values else 0.0

        travel = [t for t, _ in self.completed]
        wait = [w for _, w in self.completed]
        unfinished = getattr(self, "unfinished", 0)
        unfinished_time = getattr(self, "unfinished_time", 0.0)

        return {
            "throughput": len(self.completed),
            "unfinished": unfinished,
            "avg_travel_time_s": round(mean(travel), 2),
            "avg_waiting_time_s": round(mean(wait), 2),
            # Counting the unfinished at their time-so-far: a controller that
            # strands vehicles cannot hide them by never completing their trips.
            "avg_travel_time_incl_unfinished_s": round(
                (sum(travel) + unfinished_time) / max(1, len(travel) + unfinished), 2
            ),
            # Control quality uses the full queue; the visible one saturates.
            "avg_queue_veh": round(mean(self.full_queue_series or self.queue_series), 2),
            "max_queue_veh": max(self.full_queue_series or self.queue_series, default=0),
            "avg_visible_queue_veh": round(mean(self.queue_series), 2),
            "max_visible_queue_veh": max(self.queue_series, default=0),
            "avg_vehicles_in_net": round(mean(self.vehicle_series), 2),
            "decisions": self.decisions,
            "switches": self.switches,
            "switch_rate": round(self.switches / self.decisions, 3) if self.decisions else 0.0,
            "keep_rate": round(1 - self.switches / self.decisions, 3) if self.decisions else 0.0,
            "phase_counts": dict(sorted(self.phase_counts.items())),
            "queue_saturated_lane_seconds": self.saturated_lane_seconds,
        }
