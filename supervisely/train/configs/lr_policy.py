# by default there is no learning rate schedule
# enable it by uncommenting one of the following examples and modify its settings
# learn more in official mmpretrain docs:
# https://mmpretrain.readthedocs.io/en/dev/advanced_guides/schedule.html

# ***********************************************
# Configs examples (PLEASE, KEEP IT IN ONE LINE):
# ***********************************************

# param_scheduler = dict(type='ConstantLR', factor=1.0)
# param_scheduler = [dict(type='MultiStepLR', by_epoch=True, milestones=[30, 60, 90], gamma=0.1)]
# param_scheduler = dict(type='MultiStepLR', by_epoch=True, milestones=[100, 150], gamma=0.1)
# param_scheduler = dict(type='CosineAnnealingLR', by_epoch=True, T_max=num_epochs)
# param_scheduler = [dict(type='LinearLR', start_factor=0.001, by_epoch=False, end=50), dict(type='MultiStepLR', by_epoch=True, milestones=[8, 11], gamma=0.1)]
# param_scheduler = [dict(type='LinearLR', start_factor=0.001, by_epoch=True, end=10, convert_to_iter_based=True), dict(type='CosineAnnealingLR', by_epoch=True, begin=10)]
# param_scheduler = [dict(type='LinearLR', start_factor=0.001, by_epoch=True, end=10, convert_to_iter_based=True), dict(type='LinearMomentum', start_factor=0.001, by_epoch=False, begin=0, end=1000)]
# param_scheduler = [dict(type='CosineAnnealingLR', by_epoch=True, begin=0, end=100, eta_min=0.001)]
# param_scheduler = [dict(type='LinearLR', start_factor=0.1, by_epoch=True, begin=0, end=5, convert_to_iter_based=True), dict(type='CosineAnnealingLR', by_epoch=True, begin=5, end=100, eta_min=0, convert_to_iter_based=True)]
# param_scheduler = [dict(type='MultiStepLR', by_epoch=True, milestones=[30, 60, 90], gamma=0.1)]
# param_scheduler = [dict(type='MultiStepLR', by_epoch=True, milestones=[40, 80, 120], gamma=0.1)]
# param_scheduler = [dict(type='ExponentialLR', by_epoch=True, gamma=0.98)]
# param_scheduler = [dict(type='PolyLR', power=1.0, eta_min=0, by_epoch=False)]
# param_scheduler = [dict(type='LinearLR', start_factor=0.1, by_epoch=True, begin=0, end=5, convert_to_iter_based=True), dict(type='MultiStepLR', by_epoch=True, begin=5, end=100, milestones=[30, 60, 90], gamma=0.1)]
# param_scheduler = [dict(type='LinearLR', start_factor=0.001, by_epoch=True, begin=0, end=5, convert_to_iter_based=True), dict(type='CosineAnnealingLR', by_epoch=True, begin=5, end=100, eta_min=0, convert_to_iter_based=True)]
