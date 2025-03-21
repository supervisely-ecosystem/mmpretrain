augs_config_path = None
input_size = 224
batch_size_per_gpu = 32
num_workers_per_gpu = 2
project_dir = None

img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="SlyImgAugs", config_path=augs_config_path),
    dict(type="Resize", scale=(input_size, input_size)),
    dict(type="Normalize", **img_norm_cfg),
    dict(type='PackInputs'),
]

test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="Resize", scale=(input_size, input_size)),
    dict(type="Normalize", **img_norm_cfg),
    dict(type='PackInputs'),
]
