'''Scoring a policy: first on cached features, then inside the loop.

Public -- other subpackages read these:
    offline      trained checkpoints scored on cached features, no simulation
    closed_loop  policy inside the SUMO + Panda3D loop, batched over checkpoints
    metrics      travel time / delay / queue / throughput
    compare      every comparison method wired into the same loop

cli/:
    probe        can BEV + lane mask recover per-lane queue length at all

Offline agreement diagnoses and shortlists; it never selects. Agreeing with the
expert is not controlling well, so the ranking that counts comes from
`closed_loop` over `metrics`.
'''
