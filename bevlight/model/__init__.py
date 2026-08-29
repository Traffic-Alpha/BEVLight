'''BEV pixels -> lane -> movement -> phase -> decision.

One file per layer, each with a single job.

Public -- other subpackages read these:
    bevlight    the four layers below assembled into one model
    backbone    frozen DINOv2/v3, 1022 px -> 73x73 patch features
    mask_pool   lane mask -> patch coverage -> soft MaskPool -> v_i
    heads       auxiliary lane-state regression
    teacher     the same model reading numbers instead of pixels

_internal/ -- only this package:
    weights     where a pinned backbone lives, and which ones exist. Path
                arithmetic only, so backbone.py can ask "are the weights
                already here" without importing the downloader.

The four layers below are assembled only by bevlight.py, and their seams are
deliberate: the ablation table removes one at a time, and a layer that quietly
did two things would make those rows uninterpretable.
    temporal    frame differences + a small shared temporal transformer
    movement    lane cross-attention, f_M, movement cross-attention
    phase       permutation-invariant attention pooling over a phase's movements
    decision    per-phase shared scoring f_D + the min-green hard constraint

cli/:
    download    fetching and pinning the frozen backbone weights

No lane index, no direction label, no fixed-size output head: that is what
makes an unseen junction or an unseen signal plan just work.
'''
