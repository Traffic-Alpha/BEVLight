'''
Author: WANG Maonan
Date: 2026-04-14 14:42:24
Description: 成都光华路口配置
@LastEditTime: 2026-07-10 14:01:41
@LastEditors: WANG Maonan
'''
JUNCTION = {
    "tls_id": "INT1",
    # ===== easy 路网 =====
    "easy_low_density": {
        "num_phases": 4,
        "num_seconds": 1000,
        "fix_phase_durations": [5, 5, 5, 5],
    },
    "easy_high_density": {
        "num_phases": 4,
        "num_seconds": 1000,
        "fix_phase_durations": [5, 5, 5, 5],
    },
    "easy_fluctuating_commuter": {
        "num_phases": 4,
        "num_seconds": 1000,
        "fix_phase_durations": [5, 5, 5, 5],
    },
    "easy_increasing_demand": {
        "num_phases": 4,
        "num_seconds": 1000,
        "fix_phase_durations": [5, 5, 5, 5],
    },
    "easy_random_perturbation": {
        "num_phases": 4,
        "num_seconds": 1000,
        "fix_phase_durations": [5, 5, 5, 5],
    },
    # ===== normal 路网 =====
    "normal_low_density": {
        "num_phases": 4,
        "num_seconds": 1000,
        "fix_phase_durations": [5, 5, 5, 5],
    },
    "normal_high_density": {
        "num_phases": 4,
        "num_seconds": 1000,
        "fix_phase_durations": [5, 5, 5, 5],
    },
    "normal_fluctuating_commuter": {
        "num_phases": 4,
        "num_seconds": 1000,
        "fix_phase_durations": [5, 5, 5, 5],
    },
    "normal_increasing_demand": {
        "num_phases": 4,
        "num_seconds": 1000,
        "fix_phase_durations": [5, 5, 5, 5],
    },
    "normal_random_perturbation": {
        "num_phases": 4,
        "num_seconds": 1000,
        "fix_phase_durations": [5, 5, 5, 5],
    },
}

EVENTS = {
    "event_1": {
        "accidents": [
            {
                "id": "accident_01",
                "depart_time": 100,
                "edge_id": "-885943869#2.897",
                "lane_index": 0,
                "position": 218,
                "type": "barrier",
                "duration": 70,
            },
            {
                "id": "accident_02",
                "depart_time": 100,
                "edge_id": "-885943869#2.897",
                "lane_index": 1,
                "position": 218,
                "type": "barrier",
                "duration": 70,
            },
            {
                "id": "accident_03",
                "depart_time": 100,
                "edge_id": "-885943869#2.897",
                "lane_index": 2,
                "position": 218,
                "type": "barrier",
                "duration": 70,
            },
        ],
        "special_vehicles": [
            {
                "id": "police_01",
                "type": "police",
                "depart_time": 75,
                "route": ["-885943869#2.897", "-885943869#0"],
            },
            {
                "id": "ambulance_02",
                "type": "emergency",
                "depart_time": 250,
                "route": ["885943869#0.263", "885943869#2"],
            },
        ],
    },
}
