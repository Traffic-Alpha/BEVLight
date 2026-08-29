"""
Scenario configuration loader.

Each junction is stored as (data only, no code):

    scenarios/<junction>/
        config.py
        networks/
        routes/
        add/
        *.sumocfg

Environment names follow the pattern "{plan}_{demand}", for example
"easy_low_density" or "normal_random_perturbation".
"""

from __future__ import annotations

import importlib.util
from types import ModuleType

from ..paths import SCENARIOS_ROOT

AVAILABLE_JUNCTIONS = sorted(
    path.name
    for path in SCENARIOS_ROOT.iterdir()
    if path.is_dir() and (path / "config.py").is_file()
)


def _load_config_module(junction_name: str) -> ModuleType:
    if junction_name not in AVAILABLE_JUNCTIONS:
        raise ValueError(
            f"Unknown junction '{junction_name}'. "
            f"Available junctions: {AVAILABLE_JUNCTIONS}"
        )

    config_path = SCENARIOS_ROOT / junction_name / "config.py"
    spec = importlib.util.spec_from_file_location(
        f"scenarios.{junction_name}.config",
        config_path,
    )
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot load scenario config: {config_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_junction_config(junction_name: str, env_name: str) -> dict:
    """Load a junction environment config.

    Args:
        junction_name: Junction directory name, such as "Beijing_Beihuan".
        env_name: Environment name, such as "easy_low_density".

    Returns:
        A dictionary with SUMO paths and signal-control parameters.
    """

    junc_cfg = _load_config_module(junction_name).JUNCTION
    tls_id = junc_cfg["tls_id"]

    if env_name not in junc_cfg:
        available_envs = [key for key in junc_cfg if key != "tls_id"]
        raise ValueError(
            f"Junction '{junction_name}' has no environment '{env_name}'. "
            f"Available environments: {available_envs}"
        )

    env_cfg = junc_cfg[env_name]
    plan_name = env_name.split("_", 1)[0]
    scenario_dir = SCENARIOS_ROOT / junction_name

    return {
        "sumo_cfg": str(scenario_dir / f"{env_name}.sumocfg"),
        "net_file": str(scenario_dir / "networks" / f"{plan_name}.net.xml"),
        "tls_id": tls_id,
        "num_phases": env_cfg["num_phases"],
        "num_seconds": env_cfg["num_seconds"],
        "fix_phase_durations": env_cfg.get("fix_phase_durations"),
    }


def available_plans(junction_name: str) -> list[str]:
    """Signal plans a junction ships, read from its config env names.

    A plan is the prefix of an env name ("easy_low_density" -> "easy"), and it
    selects both `networks/<plan>.net.xml` and the phase composition inside it.
    """
    junc_cfg = _load_config_module(junction_name).JUNCTION
    plans: list[str] = []
    for key in junc_cfg:
        if key == "tls_id" or "_" not in key:
            continue
        plan = key.split("_", 1)[0]
        if plan not in plans:
            plans.append(plan)
    return plans


def available_demands(junction_name: str, plan: str) -> list[str]:
    """Demand patterns available for one plan of a junction."""
    junc_cfg = _load_config_module(junction_name).JUNCTION
    prefix = f"{plan}_"
    return [key[len(prefix):] for key in junc_cfg if key.startswith(prefix)]


def load_event_config(junction_name: str, event_name: str) -> tuple[list, list]:
    """Load accident and special-vehicle event configs for a junction."""

    config_module = _load_config_module(junction_name)
    events = getattr(config_module, "EVENTS", {})

    if event_name not in events:
        raise ValueError(
            f"Junction '{junction_name}' has no event set '{event_name}'. "
            f"Available event sets: {list(events)}"
        )

    event_cfg = events[event_name]
    return event_cfg.get("accidents", []), event_cfg.get("special_vehicles", [])
