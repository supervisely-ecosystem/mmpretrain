# Start point to search module for all registries
default_scope = "mmpretrain"

ckpt_interval = 1
log_interval = 10

# yapf:disable
default_hooks = dict(
    # Iteration time measurement and ETA calculation
    timer=dict(type='IterTimerHook'),
    # Optimizer parameters management (learning rate and etc.)
    param_scheduler=dict(type='ParamSchedulerHook'),
    # Save checkpoints
    checkpoint=dict(type='CheckpointHook', interval=ckpt_interval),
    # Main logger for Supervisely
    supervisely_logger=dict(type='SuperviselyLoggerHook', interval=log_interval)
)
# yapf:enable

dist_params = dict(backend="nccl")
log_level = "INFO"
load_from = None
resume_from = None
workflow = [("train", 1)]
classification_mode = 'one_label'
