'''
Author: WANG Maonan
Date: 2025-07-17 13:01:58
LastEditors: WMN7 18811371255@163.com
Description: 车辆 Route 生成
LastEditTime: 2026-04-13 16:16:36
'''
from tshub.utils.init_log import set_logger
from tshub.utils.get_abs_path import get_abs_path
from tshub.sumo_tools.generate_routes import generate_route

# 初始化日志
current_file_path = get_abs_path(__file__)
set_logger(current_file_path('./'), file_log_level='WARNING', terminal_log_level='INFO')

# 开启仿真 --> 指定 net 文件
sumo_net = current_file_path("./networks/normal.net.xml")

# =============================================================================
# 设定流量前必读：两条约束决定了流量的上限（现有配置已经满足，改动前请先读完）
#
# 结论先行：现有流量是标定好的，不要随便往上加。
#   scale=1.0、MaxPressure（生成标签的专家）下，六个路口实测几乎一致：
#     受控车道零占比 50~60%，p90 排队 3 辆，最大 5~8 辆，**超出画面 0.00%**
#   而同一份 route 在 FixedTime 下 p90 顶到 8、48% 的时间超出画面。
#   也就是说：坏控制器被压满（控制问题非平凡），专家的排队正好铺满 BEV 能表达
#   的 0~8 全量程且不截断（标签质量最好）。这正是想要的状态。
#
# 1) 排队太长就观测不到了。
#    模型输入是 BEV 图，进口道只露出约 60m ≈ 8 辆车（见 docs/lane_masks.md）。
#    排队顶到画面边缘后标签就被截断成下界，再堵下去对训练毫无价值，只会教模型
#    低估。France_Massy / Hongkong_YMT 的受控进口道本身只有 59m / 46m（存
#    6~8 辆），比窗口还短，那里排队会直接溢出车道；其余 10 个路口能存 22~27
#    辆，是窗口的 3 倍，所以先到极限的是画面而不是车道。
#
#    => 判据不是"饱和度 x"，而是**实测排队分布**：专家下最大排队接近但不超过
#       该路口的可见容量（约 8 辆），超窗口时间占比接近 0。
#
# 2) 不能用道路通行能力反推流量，因为排队由信号算法决定。
#    车道饱和流率(~1800 veh/h)是自由流属性；一个转向真正能放掉多少是
#    「绿信比 x 饱和流率」，而绿信比由控制器在运行时决定。自适应控制器
#    (MaxPressure) 把绿灯给到需求大的转向，其有效通行能力是算法的函数；
#    固定配时不管需求都按 1/K 分配，同样流量下排队明显更长。
#    实测 Beijing_Pinganli，同一份 route、同一个 scale=1.0：
#        MaxPressure  p90 排队 3.0 辆，超出画面  0.0% 的时间
#        FixedTime    p90 排队 8.0 辆（顶满），超出画面 47.8% 的时间
#    同一份流量对专家和对基线是完全不同的两个工况，任何"按通行能力算出来的
#    流量"都没有意义。
#
#    => 真要调，用 tools/calibrate_demand.py 去**测**，不要用公式算：
#         python tools/calibrate_demand.py --junction <名字> \
#                --scale 1.0 1.2 1.5 --controller max_pressure fixed_time
#       它用 SUMO 的 --scale 扫描，不必重新生成 route 就能定出倍数。以生成数据
#       的专家(MaxPressure)为准标定，同时看最弱基线还剩多少余量——对比实验里
#       所有方法共用这一份 route。
#
# 注意两个容易把自己骗过去的测量坑（我踩过）：
#   * 不要把几档需求平均起来看零占比：low_density 本来就有 90% 是零。
#   * 不要把 episode 的空尾巴算进去：发车 600s 就停，跑到 1000s 的话约 37%
#     是排空后的空帧，会把零占比从 55% 抬到 69%。
# =============================================================================

traffic_flow_configs = {
    # 1. 稳定低密度 (X~0.35)
    "low_density": {
        '102454134#0': [5, 5, 6, 7, 3],
        '1200878753#0': [5, 9, 7, 8, 8],
        '30658263#0': [8, 5, 7, 9, 6],
        '960661806#0': [7, 8, 9, 5, 7],
    },

    # 2. 波动通勤 (时段错峰, X~0.55)
    "fluctuating_commuter": {
        '102454134#0': [5, 7, 9, 12, 13],
        '1200878753#0': [6, 9, 15, 9, 6],
        '30658263#0': [11, 11, 12, 12, 10],
        '960661806#0': [16, 14, 11, 8, 6],
    },

    # 3. 饱和高密度 (最难但可解, X~0.72)
    "high_density": {
        '102454134#0': [20, 14, 21, 15, 18],
        '1200878753#0': [15, 21, 17, 15, 14],
        '30658263#0': [18, 12, 17, 20, 12],
        '960661806#0': [20, 15, 21, 14, 15],
    },

    # 4. 随机扰动 (随机尖峰, X~0.68)
    "random_perturbation": {
        '102454134#0': [9, 9, 10, 9, 20],
        '1200878753#0': [10, 22, 9, 9, 9],
        '30658263#0': [11, 10, 10, 23, 10],
        '960661806#0': [22, 9, 9, 9, 9],
    },

    # 5. 递增需求 (主干道主导+递增, 支路≈主向40%)
    "increasing_demand": {
        '102454134#0': [15, 20, 24, 29, 32],
        '1200878753#0': [13, 18, 21, 27, 29],
        '30658263#0': [7, 9, 9, 12, 13],
        '960661806#0': [7, 8, 9, 9, 12],
    },

}

for config_id, config_info in traffic_flow_configs.items():
    generate_route(
        sumo_net=sumo_net,
        interval=[2,2,2,2,2], # 共有 10 min
        edge_flow_per_minute=config_info,
        edge_turndef={},
        veh_type={
            'background': {'color':'220,220,220', 'length': 5, 'probability':1},
        },
        output_trip=current_file_path('./testflow.trip.xml'),
        output_turndef=current_file_path('./testflow.turndefs.xml'),
        output_route=current_file_path(f'./routes/{config_id}.rou.xml'),
        interpolate_flow=False,
        interpolate_turndef=False,
    )
