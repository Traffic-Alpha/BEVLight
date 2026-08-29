'''
@Author: WANG Maonan
@Date: 2026-06-01 01:11:04
@Description: 韩国松岛路口配置
@LastEditTime: 2026-07-10 14:01:50
@LastEditors: WANG Maonan
'''
JUNCTION = {
    "tls_id": "J2",
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
                "edge_id": "989312046#0",
                "lane_index": 1,
                "position": 215,
                "type": "barrier",
                "duration": 70,
            },
            {
                "id": "accident_02",
                "depart_time": 100,
                "edge_id": "989312046#0",
                "lane_index": 2,
                "position": 215,
                "type": "barrier",
                "duration": 70,
            },
            {
                "id": "accident_03",
                "depart_time": 100,
                "edge_id": "989312046#0",
                "lane_index": 3,
                "position": 215,
                "type": "barrier",
                "duration": 70,
            },
        ],
        "special_vehicles": [
            {
                "id": "police_01",
                "type": "police",
                "depart_time": 75,
                "route": ["989312046#0", "989312046#2"],
            },
            {
                "id": "ambulance_02",
                "type": "emergency",
                "depart_time": 250,
                "route": ["E1", "429770748#1"],
            },
        ],
    },
}