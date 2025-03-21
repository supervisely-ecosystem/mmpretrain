import sys

import sly_globals as g
import train_config


def init_script_arguments(state):
    sys.argv = [sys.argv[0]]
    sys.argv.append(train_config.main_config_path)
    sys.argv.extend(["--work-dir", g.checkpoints_dir])
    # sys.argv.extend(["--gpu-ids", state["gpusId"]])
