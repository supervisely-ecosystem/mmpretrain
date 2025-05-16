import os
import re

import sly_globals as g
from mmengine.config import Config
from src.ui import architectures

import supervisely as sly

model_config_name = "model_config.py"
dataset_config_name = "dataset_config.py"
schedule_config_name = "schedule_config.py"
runtime_config_name = "runtime_config.py"
main_config_name = "train_config.py"

configs_dir = os.path.join(g.artifacts_dir, "configs")
model_config_path = os.path.join(configs_dir, model_config_name)
dataset_config_path = os.path.join(configs_dir, dataset_config_name)
schedule_config_path = os.path.join(configs_dir, schedule_config_name)
runtime_config_path = os.path.join(configs_dir, runtime_config_name)
main_config_path = os.path.join(configs_dir, main_config_name)

main_config_template = f"""
_base_ = [
    './{model_config_name}', './{dataset_config_name}',
    './{schedule_config_name}', './{runtime_config_name}'
]
"""

sly.fs.mkdir(configs_dir)

def _replace_function(var_name, var_value, template, match):
    return template.format(var_name, var_value)

def generate_model_config(state):
    model_name = state["selectedModel"]
    model_info = architectures.get_model_info_by_name(model_name)
    lib_model_config_path = os.path.join(g.root_source_dir, model_info["modelConfig"])
    cfg = Config.fromfile(lib_model_config_path)
    with open(lib_model_config_path) as f:
        py_config = f.read()
    if state["cls_mode"] == "multi_label":
        head_name = cfg.model.head.type
        if cfg.model.head.type == "ClsHead":
            py_config = py_config.replace(
                "type='ClsHead'", "type='MultiLabelLinearClsHead'"
            )
        elif cfg.model.head.type == "CSRAClsHead":
            py_config = py_config.replace(
                "type='CSRAClsHead'", "type='CSRAHead'"
            )
        elif cfg.model.head.type == "VisionTransformerClsHead":
            head_name = "MultiLabelLinearClsHead"
            sly.logger.warning(
                "ViT models don't support multi-label classification. The common MultiLabelLinearClsHead module will be used instead."
            )

        py_config = re.sub(
            r"(head=dict\(\n\s*type)=['\"]?(\w*)['\"]?",
            lambda m: _replace_function("head=dict(type", head_name, "{}='{}'", m),
            py_config,
            flags=re.MULTILINE,
        )
        py_config = re.sub(r"topk=\(\d+,\s*\d*\),\n\s*", "", py_config, 0, re.MULTILINE)

    num_tags = len(state["selectedTags"])
    py_config = re.sub(
        r"num_classes*=(\d+)",
        lambda m: _replace_function("num_classes", num_tags, "{}={}", m),
        py_config,
        0,
        re.MULTILINE,
    )

    with open(model_config_path, "w") as f:
        f.write(py_config)
    return model_config_path, py_config


def generate_dataset_config(state):
    import ui.augs as augs

    config_path = os.path.join(g.root_source_dir, "supervisely/train/configs/dataset.py")
    if augs.augs_config_path is None:
        config_path = os.path.join(
            g.root_source_dir, "supervisely/train/configs/dataset_no_augs.py"
        )
    with open(config_path) as f:
        py_config = f.read()

    if augs.augs_config_path is not None:
        py_config = re.sub(
            r"augs_config_path\s*=\s*['\"]?(None|[\w\/\.]*)['\"]?",
            lambda m: _replace_function("augs_config_path", augs.augs_config_path, "{} = '{}'", m),
            py_config,
            flags=re.MULTILINE,
        )

    py_config = re.sub(
        r"input_size\s*=\s*(\d+)",
        lambda m: _replace_function("input_size", state["imgSize"], "{} = {}", m),
        py_config,
        0,
        re.MULTILINE,
    )

    py_config = re.sub(
        r"batch_size_per_gpu\s*=\s*(\d+)",
        lambda m: _replace_function("batch_size_per_gpu", state["batchSizePerGPU"], "{} = {}", m),
        py_config,
        0,
        re.MULTILINE,
    )

    py_config = re.sub(
        r"num_workers_per_gpu\s*=\s*(\d+)",
        lambda m: _replace_function("num_workers_per_gpu", state["workersPerGPU"], "{} = {}", m),
        py_config,
        0,
        re.MULTILINE,
    )

    py_config = re.sub(
        r"validation_interval\s*=\s*(\d+)",
        lambda m: _replace_function("validation_interval", state["valInterval"], "{} = {}", m),
        py_config,
        0,
        re.MULTILINE,
    )

    save_best = None if state["saveBest"] is False else "'auto'"
    py_config = re.sub(
        r"save_best\s*=\s*['\"]?([a-zA-Z]+)['\"]?\s",
        lambda m: _replace_function("save_best", save_best, "{} = {}\n", m),
        py_config,
        flags=re.MULTILINE,
    )

    py_config = re.sub(
        r"project_dir\s*=\s*['\"]?(None|[\w\/\.]*)['\"]?",
        lambda m: _replace_function("project_dir", g.project_dir, "{} = '{}'", m),
        py_config,
        flags=re.MULTILINE,
    )

    if state["cls_mode"] == "multi_label":
        ds_name = "SuperviselyMultiLabel"
        py_config = re.sub(
            r"dataset_type\s*=\s*['\"]?(\w*)['\"]?",
            lambda m: _replace_function("dataset_type", ds_name, "{} = '{}'", m),
            py_config,
            flags=re.MULTILINE,
        )
    else:
        ds_name = "SuperviselySingleLabel"

    train_dataloader = f"""
train_dataloader = dict(
    batch_size={state['batchSizePerGPU']},
    num_workers={state['workersPerGPU']},
    dataset=dict(
        type='{ds_name}',
        project_dir='{g.project_dir}',
        data_prefix='train',
        pipeline=train_pipeline
    ),
    sampler=dict(type='DefaultSampler', shuffle=True)
)
"""
    py_config += train_dataloader

    val_dataloader = f"""
val_dataloader = dict(
    batch_size={state['batchSizePerGPU']},
    num_workers={state['workersPerGPU']},
    dataset=dict(
        type='{ds_name}',
        project_dir='{g.project_dir}',
        data_prefix='val',
        pipeline=test_pipeline
    ),
    sampler=dict(type='DefaultSampler', shuffle=False)
)
"""
    py_config += val_dataloader

    test_dataloader = f"""
test_dataloader = dict(
    batch_size={state['batchSizePerGPU']},
    num_workers={state['workersPerGPU']},
    dataset=dict(
        type='{ds_name}',
        project_dir='{g.project_dir}',
        data_prefix='test',
        pipeline=test_pipeline
    ),
    sampler=dict(type='DefaultSampler', shuffle=False)
)
"""
    py_config += test_dataloader

    num_tags = len(state["selectedTags"])
    eval_topk = (1,5) if num_tags >= 5 else (1,)
    
    sly.logger.info(f"Number of tags: {num_tags}, using topk={eval_topk}")
    
    if state["cls_mode"] == "multi_label":
        evaluator = f"dict(type='MultiLabelMetric', average='macro')"
    else:
        evaluator = f"dict(type='SingleLabelMetric', average='macro')"

    data_preprocessor = f"""
data_preprocessor = dict(
    num_classes={len(state['selectedTags'])},
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True, # from BGR in cv2 to RGB in PIL
)
"""
    py_config += data_preprocessor

    val_cfg = "val_cfg = dict(type='ValLoop')"
    val_evaluator = f"val_evaluator = {evaluator}"
    test_cfg = "test_cfg = dict(type='TestLoop')"
    test_evaluator = f"test_evaluator = {evaluator}"

    py_config += (
        os.linesep
        + val_cfg
        + os.linesep
        + val_evaluator
        + os.linesep
        + test_cfg
        + os.linesep
        + test_evaluator
    )

    with open(dataset_config_path, "w") as f:
        f.write(py_config)
    return dataset_config_path, py_config


def generate_schedule_config(state):
    optim_wrapper = "optim_wrapper = dict(\n"
    optim_wrapper += "    optimizer=dict(\n"
    optim_wrapper += f"        type='{state['optimizer']}',\n"
    optim_wrapper += f"        lr={state['lr']},\n"
    if state["optimizer"] == "SGD" and "momentum" in state and state["momentum"] is not None:
        optim_wrapper += f"        momentum={state['momentum']},\n"
    optim_wrapper += f"        weight_decay={state['weightDecay']},\n"
    if state["optimizer"] == "SGD" and state.get("nesterov", False):
        optim_wrapper += "        nesterov=True,\n"
    optim_wrapper += "    ),\n"
    if state["gradClipEnabled"] is True:
        optim_wrapper += f"    clip_grad=dict(max_norm={state['maxNorm']}),\n"
    else:
        optim_wrapper += "    clip_grad=None,\n"
    optim_wrapper += ")\n"

    param_scheduler = ""
    if state["lrPolicyEnabled"] is True:
        py_text = state["lrPolicyPyConfig"]
        py_lines = py_text.splitlines()
        num_uncommented = 0
        for line in py_lines:
            res_line = line.strip()
            if res_line != "" and res_line[0] != "#":
                param_scheduler += res_line
                num_uncommented += 1
        if num_uncommented == 0:
            raise ValueError(
                "LR policy is enabled but not defined, please uncomment and modify one of the provided examples"
            )
        if num_uncommented > 1:
            raise ValueError("Several LR policies were uncommented, please keep only one")
    if param_scheduler == "":
        param_scheduler = "param_scheduler = dict(type='ConstantLR', factor=1.0)"

    train_cfg = f"train_cfg = dict(type='EpochBasedTrainLoop', max_epochs={state['epochs']})"
    py_config = optim_wrapper + os.linesep + param_scheduler + os.linesep + train_cfg + os.linesep

    with open(schedule_config_path, "w") as f:
        f.write(py_config)
    return schedule_config_path, py_config


def generate_runtime_config(state):
    config_path = os.path.join(g.root_source_dir, "supervisely/train/configs/runtime.py")
    with open(config_path) as f:
        py_config = f.read()

    sly.logger.info(f"Checkpoint interval: {state.get('checkpointInterval', 1)}")
    sly.logger.info(f"Save best: {state.get('saveBest', False)}")
    
    py_config = re.sub(
        r"ckpt_interval\s*=\s*(\d+)",
        lambda m: _replace_function("ckpt_interval", state["checkpointInterval"], "{} = {}", m),
        py_config,
        0,
        re.MULTILINE,
    )
    
    checkpoint_args = []
    if state.get('saveBest', False):
        checkpoint_args.append("save_best='auto'")
    
    if state.get('maxKeepCkptsEnabled', False):
        checkpoint_args.append(f"max_keep_ckpts={state.get('maxKeepCkpts', 5)}")
    
    if state.get('saveLast', False):
        checkpoint_args.append("save_last=True")
    
    if checkpoint_args:
        old_checkpoint = f"checkpoint_config = dict(interval=ckpt_interval, {', '.join(checkpoint_args)})"
        py_config = re.sub(
            r"checkpoint_config\s*=\s*dict\(interval=ckpt_interval(?:[^)]*)\)",
            lambda m: old_checkpoint,
            py_config,
            0,
            re.MULTILINE,
        )
        
        new_checkpoint = f"checkpoint=dict(type='CheckpointHook', interval=ckpt_interval, {', '.join(checkpoint_args)})"
        py_config = re.sub(
            r"checkpoint=dict\(type='CheckpointHook', interval=ckpt_interval(?:[^)]*)\)",
            lambda m: new_checkpoint,
            py_config,
            0,
            re.MULTILINE,
        )
    
    py_config = re.sub(
        r"log_interval\s*=\s*(\d+)",
        lambda m: _replace_function("log_interval", state["metricsPeriod"], "{} = {}", m),
        py_config,
        0,
        re.MULTILINE,
    )

    py_config = re.sub(
        r"load_from\s*=\s*['\"]?(None|[\w\/:\.\-]*)['\"]?",
        lambda m: _replace_function("load_from", g.local_weights_path, "{} = '{}'", m),
        py_config,
        flags=re.MULTILINE,
    )

    py_config = re.sub(
        r"classification_mode\s*=\s*['\"]?(\w*)['\"]?",
        lambda m: _replace_function("classification_mode", state["cls_mode"], "{} = '{}'", m),
        py_config,
        flags=re.MULTILINE,
    )

    with open(runtime_config_path, "w") as f:
        f.write(py_config)
    return runtime_config_path, py_config

def generate_main_config(state):
    with open(main_config_path, "w") as f:
        f.write(main_config_template)
    return main_config_path, str(main_config_template)

def save_from_state(state):
    with open(model_config_path, "w") as f:
        f.write(state["modelPyConfig"])
    with open(dataset_config_path, "w") as f:
        f.write(state["datasetPyConfig"])
    with open(schedule_config_path, "w") as f:
        f.write(state["schedulePyConfig"])
    with open(runtime_config_path, "w") as f:
        f.write(state["runtimePyConfig"])
    with open(main_config_path, "w") as f:
        f.write(state["mainPyConfig"])
