'''
@Author: WANG Maonan
@Date: 2026-06-01 01:11:04
@Description: 法国 Massy 路口配置
@LastEditTime: 2026-07-10 14:01:13
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
                "edge_id": "-172801188#1.174",
                "lane_index": 0,
                "position": 97,
                "type": "barrier",
                "duration": 70,
            },
            {
                "id": "accident_02",
                "depart_time": 100,
                "edge_id": "-172801188#1.174",
                "lane_index": 1,
                "position": 97,
                "type": "barrier",
                "duration": 70,
            },
        ],
        "special_vehicles": [
            {
                "id": "police_01",
                "type": "police",
                "depart_time": 75,
                "route": ["-172801188#1.174", "-172801188#0"],
            },
            {
                "id": "ambulance_02",
                "type": "emergency",
                "depart_time": 250,
                "route": ["172801130", "-172801188#0"],
            },
        ],
    },
}