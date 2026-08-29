'''
@Author: WANG Maonan
@Date: 2026-06-01 01:11:04
@Description: 北京北沙河路口配置
@LastEditTime: 2026-07-10 14:00:48
@LastEditors: WANG Maonan
'''
JUNCTION = {
    "tls_id": "INT1",
    # ===== easy 路网 =====
    "easy_low_density": {
        "num_phases": 3,
        "num_seconds": 1000,
        "fix_phase_durations": [5, 5, 5],
    },
    "easy_high_density": {
        "num_phases": 3,
        "num_seconds": 1000,
        "fix_phase_durations": [5, 5, 5],
    },
    "easy_fluctuating_commuter": {
        "num_phases": 3,
        "num_seconds": 1000,
        "fix_phase_durations": [5, 5, 5],
    },
    "easy_increasing_demand": {
        "num_phases": 3,
        "num_seconds": 1000,
        "fix_phase_durations": [5, 5, 5],
    },
    "easy_random_perturbation": {
        "num_phases": 3,
        "num_seconds": 1000,
        "fix_phase_durations": [5, 5, 5],
    },
    # ===== normal 路网 =====
    "normal_low_density": {
        "num_phases": 3,
        "num_seconds": 1000,
        "fix_phase_durations": [5, 5, 5],
    },
    "normal_high_density": {
        "num_phases": 3,
        "num_seconds": 1000,
        "fix_phase_durations": [5, 5, 5],
    },
    "normal_fluctuating_commuter": {
        "num_phases": 3,
        "num_seconds": 1000,
        "fix_phase_durations": [5, 5, 5],
    },
    "normal_increasing_demand": {
        "num_phases": 3,
        "num_seconds": 1000,
        "fix_phase_durations": [5, 5, 5],
    },
    "normal_random_perturbation": {
        "num_phases": 3,
        "num_seconds": 1000,
        "fix_phase_durations": [5, 5, 5],
    },
}

EVENTS = {
    "event_1": {
        "accidents": [
            {
                "id": "accident_01",
                "depart_time": 100,
                "edge_id": "657921289.337",
                "lane_index": 1,
                "position": 183,
                "type": "barrier",
                "duration": 70,
            },
            {
                "id": "accident_02",
                "depart_time": 100,
                "edge_id": "657921289.337",
                "lane_index": 2,
                "position": 183,
                "type": "barrier",
                "duration": 70,
            },
        ],
        "special_vehicles": [
            {
                "id": "police_01",
                "type": "police",
                "depart_time": 100,
                "route": ["741602130.216", "741602121"],
            },
            {
                "id": "ambulance_02",
                "type": "emergency",
                "depart_time": 300,
                "route": ["657921284.293", "741602121"],
            },
        ],
    },
}