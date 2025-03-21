import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import sly_globals as g
import sly_imgaugs
import sly_logger_hook
import sly_dataset
from dotenv import load_dotenv
from src.ui import architectures, ui

import supervisely as sly


def main():
    sly.logger.info(
        "Script arguments",
        extra={
            "context.teamId": g.team_id,
            "context.workspaceId": g.workspace_id,
            "modal.state.slyProjectId": g.project_id,
        },
    )

    g.my_app.compile_template(g.root_source_dir)

    data = {}
    state = {}
    data["taskId"] = g.task_id

    ui.init(data, state)  # init data for UI widgets
    g.my_app.run(data=data, state=state)


if __name__ == "__main__":
    sly.main_wrapper("main", main)
