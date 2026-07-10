import os
import shutil
import uuid

import supervisely as sly


def prepare_project_dir(project_dir):
    base_dir = os.path.dirname(project_dir)
    temp_dir = os.path.join(base_dir, f".{os.path.basename(project_dir)}.{uuid.uuid4().hex}.tmp")
    sly.fs.remove_dir(temp_dir)
    sly.fs.mkdir(temp_dir)
    return temp_dir


def publish_project_dir(temp_dir, project_dir):
    sly.fs.remove_dir(project_dir)
    shutil.move(temp_dir, project_dir)


def cleanup_project_dir(temp_dir):
    if temp_dir is not None:
        sly.fs.remove_dir(temp_dir)
