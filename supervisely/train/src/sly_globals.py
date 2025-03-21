import os
import sys
from pathlib import Path

import sly_functions as func
from supervisely.app.v1.app_service import AppService
from supervisely.nn.artifacts.mmclassification import MMClassification

import supervisely as sly

root_source_dir = str(Path(sys.argv[0]).parents[3])
sly.logger.info(f"Root source directory: {root_source_dir}")
sys.path.append(root_source_dir)
source_path = str(Path(sys.argv[0]).parents[0])
sly.logger.info(f"App source directory: {source_path}")
sys.path.append(source_path)
ui_sources_dir = os.path.join(source_path, "ui")
sly.logger.info(f"UI source directory: {ui_sources_dir}")
sys.path.append(ui_sources_dir)
sly.logger.info(f"Added to sys.path: {ui_sources_dir}")

# @TODO: for debug
from dotenv import load_dotenv

if sly.is_development():
    sly_env_path = os.path.join(root_source_dir, "supervisely", "train", "supervisely.env")
    load_dotenv(sly_env_path)
    debug_env_path = os.path.join(root_source_dir, "supervisely", "train", "local.env")
    load_dotenv(debug_env_path)

my_app = AppService()
api = my_app.public_api
task_id = my_app.task_id

team_id = sly.env.team_id()
workspace_id = sly.env.workspace_id()
project_id = sly.env.project_id()

sly_mmcls = MMClassification(team_id)
project_stats = api.project.get_stats(project_id)
project_info = api.project.get_info_by_id(project_id)
if project_info is None:  # for debug
    raise ValueError(f"Project with id={project_id} not found")

# sly.fs.clean_dir(my_app.data_dir)  # @TODO: for debug

project_dir = os.path.join(my_app.data_dir, "sly_project")
project_meta = sly.ProjectMeta.from_json(api.project.get_meta(project_id))
project_meta_json = project_meta.to_json()

image_ids = [image_info.id for image_info in api.image.get_list(project_id=project_info.id)]
images_infos = None
my_app.logger.info("Image ids are initialized", extra={"count": len(image_ids)})

data_dir = sly.app.get_synced_data_dir()
artifacts_dir = os.path.join(data_dir, "artifacts")
sly.fs.mkdir(artifacts_dir)
info_dir = os.path.join(artifacts_dir, "info")
sly.fs.mkdir(info_dir)
checkpoints_dir = os.path.join(artifacts_dir, "checkpoints")
sly.fs.mkdir(checkpoints_dir)

sly_mmdet_generated_metadata = None  # for project Workflow purposes

cls_mode = "one_label"
devices = func.get_gpu_devices()
local_weights_path = None
