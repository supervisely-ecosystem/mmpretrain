import os
import cv2
import numpy as np
import torch
from torchvision import transforms
from mmengine.config import Config
from mmengine.runner import load_checkpoint

from mmpretrain.structures import DataSample
from mmpretrain.registry import MODELS


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
    
    model = MODELS.build(cfg.model)
    checkpoint = load_checkpoint(model, g.local_weights_path, map_location=g.torch_device)
    model = _load_classes(model, checkpoint)
    model = model.to(g.torch_device)
    model.eval()
    g.model = model
    sly.logger.info("🟩 Model has been successfully deployed")


def perform_inference(model, img, topn=5):
    input_tensor = _preprocess_image(img)
    results, is_datasample = _predict_with_model(model, input_tensor)
    if is_datasample:
        return _process_datasample_results(results, topn)
    else:
        return _process_tensor_results(results, topn)


def perform_inference_batch(model, images_nps, topn=5):
    inference_results = []
    for images_batch in sly.batched(images_nps, g.batch_size):
        input_tensors = torch.cat([_preprocess_image(img) for img in images_batch], dim=0)
        data_samples = [DataSample().to(g.torch_device) for _ in range(len(images_batch))]
        results, is_datasample = _predict_with_model_batch(model, input_tensors, data_samples)
        batch_results = _process_batch_results(results, is_datasample, topn)
        inference_results.extend(batch_results)
    return inference_results

def _predict_with_model(model, input_tensor):
    with torch.no_grad():
        data_samples = [DataSample().to(g.torch_device)]
        results = model(input_tensor, data_samples=data_samples, mode='predict')
        is_datasample = isinstance(results, list) and isinstance(results[0], DataSample)
        return results, is_datasample
    
def _load_classes(model, checkpoint):
    if 'meta' in checkpoint and 'CLASSES' in checkpoint['meta']:
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = sorted(g.gt_labels, key=g.gt_labels.get)
    return model


def _preprocess_image(img):
    if isinstance(img, str):
        image = cv2.imread(img)
        if image is None:
            raise ValueError(f"Failed to load image from path: {img}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image = img
    
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return preprocess(image).unsqueeze(0).to(g.torch_device)


def _process_datasample_results(sample, topn=5):
    if hasattr(sample, 'pred_score'):
        scores = sample.pred_score.cpu().numpy()
        if topn is None:  # multi-label
            indices = np.where(scores > 0.5)[0]
            for idx in indices:
                class_name = g.gt_index_to_labels.get(idx, None)
                return {
                    "label": [int(idx)],
                    "score": [float(scores[idx])],
                    "class": [class_name]
                }
        else:  # single-label top-n
            top_indices = scores.argsort()[-topn:][::-1]
            for idx in top_indices:
                class_name = g.gt_index_to_labels.get(idx, None)
                return {
                    "label": [int(idx)],
                    "score": [float(scores[idx])],
                    "class": [class_name]
                }


def _process_tensor_results(results, topn=5):
    scores = results[0].cpu().numpy() if isinstance(results[0], torch.Tensor) else results[0]
    output_list = []
    
    if topn is None:  # multi-label
        indices = np.where(scores > 0.5)[0]
        for idx in indices:
            class_name = g.gt_index_to_labels.get(idx, None)
            output_list.append({
                "label": int(idx),
                "score": float(scores[idx]),
                "class": class_name
            })
    else:  # single-label top-n
        top_indices = scores.argsort()[-topn:][::-1]
        for idx in top_indices:
            class_name = g.gt_index_to_labels.get(idx, None)
            output_list.append({
                "label": int(idx),
                "score": float(scores[idx]),
                "class": class_name
            })
    
    return output_list

def _predict_with_model_batch(model, input_tensor, data_samples):
    with torch.no_grad():
        results = model(input_tensor, data_samples=data_samples, mode='predict')
        is_datasample = isinstance(results, list) and isinstance(results[0], DataSample)
        return results, is_datasample

def _process_batch_results(results, is_datasample, topn=5):
    if is_datasample:
        return [_process_datasample_results(sample, topn) for sample in results]
    else:
        if isinstance(results, torch.Tensor):
            batch_scores = results.cpu().numpy()
        elif isinstance(results, list) and isinstance(results[0], torch.Tensor):
            batch_scores = np.array([tensor.cpu().numpy() for tensor in results])
        else:
            batch_scores = np.array(results)
        batch_results = []
        for scores in batch_scores:
            single_result = [scores]
            batch_results.append(_process_tensor_results(single_result, topn))
        return batch_results