'''SUMO rollout -> one shared trajectory JSON per episode. No rendering here.

Simulation and rendering are decoupled on purpose: one trajectory can be
replayed into many appearance variants that all share the same labels.

Public -- other subpackages read these:
    observation      per-lane queue / occupancy truth, counted only over the
                     stretch of lane that is inside the BEV window
    episode_schema   the trajectory file format and its version
    frame_selection  which simulated seconds are worth a Blender render

cli/ -- one `bevlight collect ...` command each, imported by nothing else:
    collect          run and record episodes over a scenario selection
    blender          render selected frames offline with Cycles
    calibrate_demand pick a demand that lands queues in the observable band
    frame_check      mask-over-frame contact sheets for a collected episode
'''
