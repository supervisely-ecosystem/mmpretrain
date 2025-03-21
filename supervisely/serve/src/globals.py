import os
import pathlib
import sys
import torch
import supervisely.io.env as env
from supervisely.app.v1.app_service import AppService
from supervisely.nn.inference.cache import InferenceImageCache

import supervisely as sly

root_source_path = str(pathlib.Path(sys.argv[0]).parents[3])
sly.logger.info(f"Root source directory: {root_source_path}")
sys.path.append(root_source_path)

train_source_path = os.path.join(root_source_path, "supervisely/train/src")
sly.logger.info(f"Train source directory: {train_source_path}")
sys.path.append(train_source_path)

serve_source_path = os.path.join(root_source_path, "supervisely/serve/src")
sly.logger.info(f"Serve source directory: {serve_source_path}")
sys.path.append(serve_source_path)

from dotenv import load_dotenv

if sly.is_development():
    debug_env_path = os.path.join(root_source_path, "supervisely", "serve", "local.env")
    load_dotenv(debug_env_path)
    sly_env_path = os.path.join(root_source_path, "supervisely", "serve", "supervisely.env")
    load_dotenv(sly_env_path)

# if sly.is_development():
#     load_dotenv("local.env")
#     load_dotenv(os.path.expanduser("~/supervisely.env"))

my_app = AppService()
api = my_app.public_api
task_id = my_app.task_id

# sly.fs.clean_dir(my_app.data_dir)  # @TODO: for debug

team_id = sly.env.team_id()
workspace_id = sly.env.workspace_id()

device = torch.device(os.environ["modal.state.device"])
remote_weights_path = os.environ["modal.state.slyFile"]
batch_size = int(os.getenv("modal.state.batch_size", 256))

remote_exp_dir = str(pathlib.Path(remote_weights_path).parents[1])
remote_configs_dir = os.path.join(remote_exp_dir, "configs")
remote_info_dir = os.path.join(remote_exp_dir, "info")

local_weights_path = os.path.join(
    my_app.data_dir, sly.fs.get_file_name_with_ext(remote_weights_path)
)
local_configs_dir = os.path.join(my_app.data_dir, "configs")
sly.fs.mkdir(local_configs_dir)
local_model_config_path = os.path.join(local_configs_dir, "train_config.py")

local_info_dir = os.path.join(my_app.data_dir, "info")
sly.fs.mkdir(local_info_dir)
local_gt_labels_path = os.path.join(local_info_dir, "gt_labels.json")
local_labels_urls_path = os.path.join(local_info_dir, "tag2urls.json")


model = None
cfg = None

meta: sly.ProjectMeta = None
gt_labels = None  # name -> index
gt_index_to_labels = None  # index -> name
labels_urls = None
cls_mode = "one_label"
inference_requests = {}
cache = InferenceImageCache(
    maxsize=env.smart_cache_size(),
    ttl=env.smart_cache_ttl(),
    is_persistent=True,
    base_folder=env.smart_cache_container_dir(),
)
