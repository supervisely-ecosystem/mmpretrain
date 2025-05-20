import os

import sly_globals as g
import splits

import supervisely as sly


def init(data, state):
    state["epochs"] = 20
    state["gpusId"] = "0"
    state["devices"] = g.devices

    state["imgSize"] = 224
    state["batchSizePerGPU"] = 32
    state["workersPerGPU"] = 2  # @TODO: 0 - for debug
    state["valInterval"] = 1
    state["metricsPeriod"] = 10
    state["checkpointInterval"] = 1
    state["maxKeepCkptsEnabled"] = True
    state["maxKeepCkpts"] = 3
    state["saveLast"] = True
    state["saveBest"] = True
    state["disabledImgSize"] = False

    state["optimizer"] = "SGD"
    state["lr"] = 0.001
    state["momentum"] = 0.9
    state["weightDecay"] = 0.0001
    state["nesterov"] = False
    state["gradClipEnabled"] = False
    state["maxNorm"] = 1

    state["lrPolicyEnabled"] = False

    file_path = os.path.join(g.root_source_dir, "supervisely/train/configs/lr_policy.py")
    with open(file_path) as f:
        state["lrPolicyPyConfig"] = f.read()

    state["collapsed7"] = True
    state["disabled7"] = True
    data["done7"] = False


def restart(data, state):
    data["done7"] = False


@g.my_app.callback("use_hyp")
@sly.timeit
@g.my_app.ignore_errors_and_show_dialog_window()
def use_hyp(api: sly.Api, task_id, context, state, app_logger):
    from torch import cuda

    gpu_id = state["gpusId"]
    device_str = "cuda:" + gpu_id
    cuda.set_device(device_str)
    app_logger.info(f"GPU device set ({device_str})", extra={"gpu_id": gpu_id})
    metric_period = state["metricsPeriod"]
    if state["batchSizePerGPU"] * state["metricsPeriod"] > state["final_train_size"]:
        metric_period = 1
    fields = [
        {"field": "data.done7", "payload": True},
        {"field": "state.metricsPeriod", "payload": metric_period},
        {"field": "state.collapsed8", "payload": False},
        {"field": "state.disabled8", "payload": False},
        {"field": "state.activeStep", "payload": 8},
    ]
    g.api.app.set_fields(g.task_id, fields)
