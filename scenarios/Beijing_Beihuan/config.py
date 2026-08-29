'''
Author: WANG Maonan
Date: 2026-04-14 14:42:21
Description: 北京北环路口配置
@LastEditTime: 2026-07-10 14:00:41
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

# 特殊事件配置: 按「事件集合」名称组织, 评估时通过 --event_name 选择
#   accidents:        指定时间在某车道位置放置静止路障 (停 duration 秒)
#   special_vehicles: 指定时间发车的特殊车辆 (救护车/警车/消防车等), 沿 route 行驶
# 注意: edge_id 需与所用路网 (easy/normal) 一致; type 若未在 scenario 中定义,
#       TSCEventWrapper 会自动从 DEFAULT_VEHTYPE 复制 (仅外观为默认样式)。
EVENTS = {
    "event_1": {
        "accidents": [
            {
                "id": "accident_01",
                "depart_time": 60,
                "edge_id": "-1106233488.150",
                "lane_index": 0,
                "position": 197.5,
                "type": "barrier",
                "duration": 70,
            },
            {
                "id": "accident_02",
                "depart_time": 60,
                "edge_id": "-1106233488.150",
                "lane_index": 1,
                "position": 197.5,
                "type": "barrier",
                "duration": 70,
            },
        ],
        "special_vehicles": [
            {
                "id": "police_01",
                "type": "police",
                "depart_time": 75,
                "route": ["-1106233488.150", "-252712271#1"],
            },
            {
                "id": "ambulance_02",
                "type": "emergency",
                "depart_time": 200,
                "route": ["157863208#0.1590", "1106233488"],
            },
            {
                "id": "fire_05",
                "type": "fire_engine",
                "depart_time": 378,
                "route": ["252712271#0.589", "1106233488"],
            },
        ],
    },
}