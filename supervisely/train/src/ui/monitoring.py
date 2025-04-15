import os

import sly_globals as g
import workflow as w
from sly_train_progress import init_progress
from dataclasses import asdict
from supervisely.io.fs import list_files, list_files_recursively, get_file_name_with_ext
from supervisely.io.json import dump_json_file
from supervisely.nn.artifacts.artifacts import TrainInfo

import supervisely as sly
from tools.train import main as mm_train

_open_lnk_name = "open_app.lnk"


def init(data, state):
    init_progress("Epoch", data)
    init_progress("Iter", data)
    init_progress("UploadDir", data)
    data["eta"] = None

    init_charts(data, state)

    state["collapsed9"] = True
    state["disabled9"] = True
    state["done9"] = False

    state["started"] = False

    data["outputName"] = None
    data["outputUrl"] = None
    state["isValidation"] = False


def init_chart(title, names, xs, ys, smoothing=None, yrange=None, decimals=None, xdecimals=None):
    series = []
    for name, x, y in zip(names, xs, ys):
        series.append({"name": name, "data": [[px, py] for px, py in zip(x, y)]})
    result = {
        "options": {
            "title": title,
            # "groupKey": "my-synced-charts",
        },
        "series": series,
    }
    if smoothing is not None:
        result["options"]["smoothingWeight"] = smoothing
    if yrange is not None:
        result["options"]["yaxisInterval"] = yrange
    if decimals is not None:
        result["options"]["decimalsInFloat"] = decimals
    if xdecimals is not None:
        result["options"]["xaxisDecimalsInFloat"] = xdecimals
    return result


def init_charts(data, state):
    # demo_x = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]
    # demo_y = [[0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001]]
    data["chartLR"] = init_chart(
        "LR", names=["LR"], xs=[[]], ys=[[]], smoothing=None, decimals=6, xdecimals=2
    )
    data["chartTrainLoss"] = init_chart(
        "Train Loss", names=["train"], xs=[[]], ys=[[]], smoothing=0.6, decimals=6, xdecimals=2
    )
    data["chartValMetrics"] = init_chart(
        "Val metrics",
        names=["precision", "recall", "F1"],
        xs=[[], [], []],
        ys=[[], [], []],
        decimals=6,
        smoothing=0.6,
    )
    data["chartTime"] = init_chart("Time", names=["time"], xs=[[]], ys=[[]], xdecimals=2)
    data["chartDataTime"] = init_chart(
        "Data Time", names=["data_time"], xs=[[]], ys=[[]], xdecimals=2
    )
    data["chartMemory"] = init_chart("Memory", names=["memory"], xs=[[]], ys=[[]], xdecimals=2)
    state["smoothing"] = 0.6


def _save_link_to_ui(local_dir, app_url):
    # save report to file *.lnk (link to report)
    local_path = os.path.join(local_dir, _open_lnk_name)
    sly.fs.ensure_base_path(local_path)
    with open(local_path, "w") as text_file:
        print(app_url, file=text_file)

from functools import partial

from sly_train_args import init_script_arguments
from sly_train_progress import _update_progress_ui


def upload_artifacts_and_log_progress():
    _save_link_to_ui(g.artifacts_dir, g.my_app.app_url)

    def upload_monitor(monitor, api: sly.Api, task_id, progress: sly.Progress):
        if progress.total == 0:
            progress.set(monitor.bytes_read, monitor.len, report=False)
        else:
            progress.set_current_value(monitor.bytes_read, report=False)
        _update_progress_ui("UploadDir", g.api, g.task_id, progress)

    model_dir = f"{g.sly_mmcls.framework_folder}-v2"  # save to '/mmclassification-v2'
    remote_artifacts_dir = f"{model_dir}/{g.task_id}_{g.project_info.name}"
    remote_weights_dir = os.path.join(remote_artifacts_dir, g.sly_mmcls.weights_folder)

    local_files = list_files(g.artifacts_dir)
    for dir in os.listdir(g.artifacts_dir):
        if dir == "checkpoints":
            local_files += list_files(os.path.join(g.artifacts_dir, dir), valid_extensions=[".pth"])
        else:
            local_files += list_files_recursively(os.path.join(g.artifacts_dir, dir))

    remote_files = [file.replace(g.artifacts_dir, remote_artifacts_dir) for file in local_files]
    total_size = sum([sly.fs.get_file_size(file_path) for file_path in local_files])

    progress = sly.Progress("Upload directory with training artifacts to Team Files", total_size, is_size=True)
    progress_cb = partial(upload_monitor, api=g.api, task_id=g.task_id, progress=progress)
    g.api.file.upload_bulk(g.team_id, local_files, remote_files, progress_cb=progress_cb)
    g.sly_mmdet_generated_metadata = g.sly_mmcls.generate_metadata(
        app_name="Train MMClassification V2 (MMPretrain)",
        task_id=g.task_id,
        artifacts_folder=remote_artifacts_dir,
        weights_folder=remote_weights_dir,
        weights_ext=g.sly_mmcls.weights_ext,
        project_name=g.project_info.name,
        task_type=g.sly_mmcls.task_type,
        config_path=None,
    )
    return remote_artifacts_dir


def create_experiment(model_name, remote_dir):
    train_info = TrainInfo(**g.sly_mmdet_generated_metadata)
    experiment_info = g.sly_mmcls.convert_train_to_experiment_info(train_info)
    experiment_info.experiment_name = f"{g.task_id}_{g.project_info.name}_{model_name}"
    experiment_info.model_name= model_name
    experiment_info.framework_name = f"{g.sly_mmcls.framework_name} V2"
    experiment_info.train_size = g.train_size
    experiment_info.val_size = g.val_size
    experiment_info_json = asdict(experiment_info)
    experiment_info_json["project_preview"] = g.project_info.image_preview_url
    g.api.task.set_output_experiment(g.task_id, experiment_info_json)
    experiment_info_json.pop("project_preview")
    
    experiment_info_path = os.path.join(g.artifacts_dir, "experiment_info.json")
    remote_experiment_info_path = os.path.join(remote_dir, "experiment_info.json")
    dump_json_file(experiment_info_json, experiment_info_path)
    g.api.file.upload(g.team_id, experiment_info_path, remote_experiment_info_path)
    

@g.my_app.callback("train")
@sly.timeit
@g.my_app.ignore_errors_and_show_dialog_window()
def train(api: sly.Api, task_id, context, state, app_logger):
    try:
        sly.json.dump_json_file(state, os.path.join(g.info_dir, "ui_state.json"))
        g.cls_mode = state["cls_mode"]

        init_script_arguments(state)
        mm_train()
        w.workflow_input(api, g.project_info, state)
        # hide progress bars and eta
        fields = [
            {"field": "data.progressEpoch", "payload": None},
            {"field": "data.progressIter", "payload": None},
            {"field": "data.eta", "payload": None},
        ]
        g.api.app.set_fields(g.task_id, fields)

        remote_dir = upload_artifacts_and_log_progress()
        
        try:
            sly.logger.info("Creating experiment info")
            create_experiment(state["selectedModel"], remote_dir)
        except Exception as e:
            sly.logger.warning(f"Couldn't create experiment, this training session will not appear in experiments table. Error: {e}")
        
        file_info = api.file.get_info_by_path(g.team_id, os.path.join(remote_dir, _open_lnk_name))
        api.task.set_output_directory(task_id, file_info.id, remote_dir)

        # show result directory in UI
        fields = [
            {"field": "data.outputUrl", "payload": g.api.file.get_url(file_info.id)},
            {"field": "data.outputName", "payload": remote_dir},
            {"field": "state.done9", "payload": True},
            {"field": "state.started", "payload": False},
        ]
        g.api.app.set_fields(g.task_id, fields)
    except Exception as e:
        api.app.set_field(task_id, "state.started", False)
        raise e  # app will handle this error and show modal window

    w.workflow_output(api, g.sly_mmdet_generated_metadata, state)
    # stop application
    g.my_app.stop()
