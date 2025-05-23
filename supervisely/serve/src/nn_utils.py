import os
from mmengine.config import Config
from mmpretrain import ImageClassificationInferencer

import supervisely as sly
import globals as g


@sly.timeit
def download_model_and_configs():
    if not g.remote_weights_path.endswith(".pth"):
        raise ValueError(
            f"Unsupported weights extension {sly.fs.get_file_ext(g.remote_weights_path)}. "
            f"Supported extension: '.pth'"
        )

    info = g.api.file.get_info_by_path(g.team_id, g.remote_weights_path)
    if info is None:
        raise FileNotFoundError(f"Weights file not found: {g.remote_weights_path}")

    progress = sly.Progress("Downloading weights", info.sizeb, is_size=True, need_info_log=True)
    g.local_weights_path = os.path.join(
        g.my_app.data_dir, sly.fs.get_file_name_with_ext(g.remote_weights_path)
    )
    g.api.file.download(
        g.team_id,
        g.remote_weights_path,
        g.local_weights_path,
        cache=g.my_app.cache,
        progress_cb=progress.iters_done_report,
    )

    def _download_dir(remote_dir, local_dir):
        remote_files = g.api.file.list2(g.team_id, remote_dir)
        progress = sly.Progress(f"Downloading {remote_dir}", len(remote_files), need_info_log=True)
        for remote_file in remote_files:
            local_file = os.path.join(local_dir, sly.fs.get_file_name_with_ext(remote_file.path))
            if sly.fs.file_exists(local_file):  # @TODO: for debug
                pass
            else:
                g.api.file.download(g.team_id, remote_file.path, local_file)
            progress.iter_done_report()

    _download_dir(g.remote_configs_dir, g.local_configs_dir)
    _download_dir(g.remote_info_dir, g.local_info_dir)

    sly.logger.info("Model has been successfully downloaded")


def construct_model_meta():
    g.labels_urls = sly.json.load_json_file(g.local_labels_urls_path)
    g.gt_labels = sly.json.load_json_file(g.local_gt_labels_path)
    g.gt_index_to_labels = {index: name for name, index in g.gt_labels.items()}

    tag_metas = []
    for name, index in g.gt_labels.items():
        tag_metas.append(sly.TagMeta(name, sly.TagValueType.NONE))
    g.meta = sly.ProjectMeta(tag_metas=sly.TagMetaCollection(tag_metas))


@sly.timeit
def deploy_model():
    cfg = Config.fromfile(g.local_model_config_path)
    g.cfg = cfg
    if hasattr(cfg, "classification_mode"):
        g.cls_mode = cfg.classification_mode

    g.inferencer = ImageClassificationInferencer(
        g.local_model_config_path,
        pretrained=g.local_weights_path,
        device=g.torch_device,
    )
    sly.logger.info("mmpretrain ImageClassificationInferencer initialized")
    sly.logger.info("🟩 Model has been successfully deployed")


def perform_inference(img, topn: int = 5):
    import numpy as np

    pred = g.inferencer(img)[0]
    scores: np.ndarray = pred["pred_scores"]

    if topn == 1:
        top_indices = [int(scores.argmax())]
    else:
        k = min(topn, scores.shape[0])
        top_indices = np.argpartition(scores, -k)[-k:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

    labels = [int(i) for i in top_indices]
    scores_list = [float(scores[i]) for i in top_indices]
    classes = [g.gt_index_to_labels.get(i, None) for i in top_indices]

    return [{
        "label": labels,
        "score": scores_list,
        "class": classes,
    }]

def perform_inference_batch(images_nps, topn: int = 5):
    import numpy as np

    predictions = g.inferencer(images_nps, batch_size=g.batch_size)
    idx2label = g.gt_index_to_labels

    results = []
    for sample in predictions:
        scores: np.ndarray = sample["pred_scores"]

        if topn == 1:
            top_indices = [int(scores.argmax())]
        else:
            k = min(topn, scores.shape[0])
            top_indices = np.argpartition(scores, -k)[-k:]
            top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        labels = [int(i) for i in top_indices]
        scores_list = [float(scores[i]) for i in top_indices]
        classes = [idx2label.get(i, None) for i in top_indices]

        results.append({
            "label": labels,
            "score": scores_list,
            "class": classes,
        })

    return results
