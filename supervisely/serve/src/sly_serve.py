import cv2
import os
import functools
from functools import lru_cache
import time
from typing import Dict, List
from concurrent.futures import ThreadPoolExecutor
import threading
import uuid

import numpy as np
import supervisely as sly
from supervisely import batched
from supervisely.task.progress import Progress

import globals as g
import functions as f
import nn_utils
import workflow as w

# Register all transform classes
from mmengine import init_default_scope
init_default_scope("mmpretrain")


@lru_cache(maxsize=10)
def get_image_by_id(image_id):
    img = g.api.image.download_np(image_id)
    return img


def send_error_data(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        value = None
        try:
            value = func(*args, **kwargs)
        except Exception as e:
            request_id = kwargs["context"]["request_id"]
            g.my_app.send_response(request_id, data={"error": repr(e)})
        return value

    return wrapper


@g.my_app.callback("get_model_meta")
@sly.timeit
@send_error_data
def get_model_meta(api: sly.Api, task_id, context, state, app_logger):
    request_id = context["request_id"]
    g.my_app.send_response(request_id, data=g.meta.to_json())


@g.my_app.callback("get_tags_examples")
@sly.timeit
@send_error_data
def get_tags_examples(api: sly.Api, task_id, context, state, app_logger):
    request_id = context["request_id"]
    g.my_app.send_response(request_id, data=g.labels_urls)


@g.my_app.callback("get_session_info")
@sly.timeit
@send_error_data
def get_session_info(api: sly.Api, task_id, context, state, app_logger):
    info = {
        "app": "MM Classification Serve",
        "weights": g.remote_weights_path,
        "device": g.device,
        "session_id": task_id,
        "classes_count": len(g.meta.tag_metas),
        "classification_mode": g.cls_mode
    }
    request_id = context["request_id"]
    g.my_app.send_response(request_id, data=info)


def inference_image_path(image_path, context, state, app_logger):
    app_logger.debug("Input path", extra={"path": image_path})
    res_path = image_path
    if "rectangle" in state:
        image = sly.image.read(image_path)  # RGB image
        top, left, bottom, right = f.get_bbox_with_padding(rectangle=state['rectangle'], pad_percent=state.get('pad', 0),
                                                           img_size=image.shape[:2])  # img_size=(h,w)

        rect = sly.Rectangle(top, left, bottom, right)
        canvas_rect = sly.Rectangle.from_size(image.shape[:2])
        results = rect.crop(canvas_rect)
        if len(results) != 1:
            return {
                "message": "roi rectangle out of image bounds",
                "roi": state["rectangle"],
                "img_size": {"height": image.shape[0], "width": image.shape[1]}
            }
        rect = results[0]
        cropped_image = sly.image.crop(image, rect)
        res_path = os.path.join(g.my_app.data_dir, sly.rand_str(10) + sly.fs.get_file_ext(image_path))
        sly.image.write(res_path, cropped_image)

    res = nn_utils.perform_inference(g.model, res_path, topn=state.get("topn", 5))
    if "rectangle" in state:
        sly.fs.silent_remove(res_path)

    return res

@g.my_app.callback("inference_image_url")
@sly.timeit
@send_error_data
def inference_image_url(api: sly.Api, task_id, context, state, app_logger):
    app_logger.debug("Input data", extra={"state": state})
    image_url = state["image_url"]
    ext = sly.fs.get_file_ext(image_url)
    if ext == "":
        ext = ".jpg"
    local_image_path = os.path.join(g.my_app.data_dir, sly.rand_str(15) + ext)
    sly.fs.download(image_url, local_image_path)
    results = inference_image_path(local_image_path, context, state, app_logger)
    sly.fs.silent_remove(local_image_path)

    request_id = context["request_id"]
    g.my_app.send_response(request_id, data=results)


@g.my_app.callback("inference_image_id")
@sly.timeit
@send_error_data
def inference_image_id(api: sly.Api, task_id, context, state, app_logger):
    sly.logger.info("infer image id", extra={"state": state})

    image_id = state["image_id"]

    image_info = api.image.get_info_by_id(image_id)
    image_path = os.path.join(g.my_app.data_dir, f"{image_id}{sly.fs.get_file_ext(image_info.name)}")
    img = get_image_by_id(image_id)
    sly.image.write(image_path, img)

    predictions = inference_image_path(image_path, context, state, app_logger)
    sly.fs.silent_remove(image_path)
    request_id = context["request_id"]
    g.my_app.send_response(request_id, data=predictions)


@g.my_app.callback("inference_batch_ids")
@sly.timeit
@send_error_data
def inference_batch_ids(api: sly.Api, task_id, context, state, app_logger):
    sly.logger.info("inference batch ids called:", extra={"state": state})

    # load images
    images_nps = f.get_nps_images(images_ids=state["images_ids"])
    images_to_process = f.crop_images(images_nps=images_nps, rectangles=state.get('rectangles'), padding=state.get('pad', 0))
    images_indexes_to_process = [index for index, img_np in enumerate(images_to_process) if img_np is not None]
    inference_results = nn_utils.perform_inference_batch(model=g.model, images_nps=images_to_process, topn=state.get('topn', 5))
    results = [None for _ in images_nps]
    for index, row in enumerate(inference_results):
        results[images_indexes_to_process[index]] = row

    g.my_app.send_response(request_id=context["request_id"], data=results)


def _inference_images_ids_async(api: sly.Api, state: Dict, inference_request_uuid: str, app_logger):
    batch_size = 16
    download_executor = ThreadPoolExecutor(batch_size)
    def _download_images(image_ids: List[int]):
        for image_id in image_ids:
            download_executor.submit(
                g.cache.download_image,
                api,
                image_id,
            )
        download_executor.shutdown(wait=True)
    download_images_thread = None
    inference_request = None
    try:
        inference_request = g.inference_requests[inference_request_uuid]
        images_ids=state["images_ids"]
        rectangles=state.get('rectangles')
        padding=state.get('pad', 0)
        topn=state.get('topn', 5)

        sly_progress: Progress = inference_request["progress"]
        sly_progress.total = len(images_ids)

        # download images
        download_images_thread = threading.Thread(target=_download_images, args=(images_ids,))
        download_images_thread.start()

        result = []
        for batch_ids in batched(images_ids, batch_size=batch_size):
            if inference_request["cancel_inference"]:
                app_logger.debug(
                    "Cancelling inference project...",
                    extra={"inference_request_uuid": inference_request_uuid},
                )
                result = []
                break
            images_nps = [g.cache.download_image(api, im_id) for im_id in batch_ids]
            images_to_process = f.crop_images(images_nps=images_nps, rectangles=rectangles, padding=padding)
            images_indexes_to_process = [index for index, img_np in enumerate(images_to_process) if img_np is not None]
            inference_results = nn_utils.perform_inference_batch(model=g.model, images_nps=images_to_process, topn=topn)

            batch_results = [None for _ in images_nps]
            for index, row in enumerate(inference_results):
                batch_results[images_indexes_to_process[index]] = row

            inference_request["pending_results"].extend(batch_results)
            result.extend(batch_results)
            sly_progress.iters_done(len(batch_ids))

        inference_request["result"] = result
    except Exception as e:
        if inference_request is not None:
            inference_request["exception"] = str(e)
        app_logger.error(f"Error in _inference_images_ids_async function: {e}", exc_info=True)
        raise
    finally:
        download_executor.shutdown(wait=False)
        if download_images_thread is not None:
            download_images_thread.join()
        if inference_request is not None:
            inference_request["is_inferring"] = False


def _on_async_inference_start(inference_request_uuid: str):
    inference_request = {
        "progress": Progress("Inferring model...", total_cnt=1),
        "is_inferring": True,
        "cancel_inference": False,
        "result": None,
        "pending_results": [],
        "preparing_progress": {"current": 0, "total": 1},
        "exception": None,
    }
    g.inference_requests[inference_request_uuid] = inference_request


@g.my_app.callback("inference_batch_ids_async")
def inference_batch_ids_async(api: sly.Api, task_id, context, state, app_logger):
    inference_request_uuid = uuid.uuid5(
        namespace=uuid.NAMESPACE_URL, name=f"{time.time()}"
    ).hex
    _on_async_inference_start(inference_request_uuid)

    threading.Thread(target=_inference_images_ids_async, args=(api, state, inference_request_uuid, app_logger)).start()

    g.my_app.send_response(context["request_id"], data={"inference_request_uuid": inference_request_uuid})


def _convert_sly_progress_to_dict(sly_progress: Progress):
    return {
        "current": sly_progress.current,
        "total": sly_progress.total,
    }


def _get_log_extra_for_inference_request(inference_request_uuid, inference_request: dict):
    log_extra = {
        "uuid": inference_request_uuid,
        "progress": inference_request["progress"],
        "is_inferring": inference_request["is_inferring"],
        "cancel_inference": inference_request["cancel_inference"],
        "has_result": inference_request["result"] is not None,
        "pending_results": len(inference_request["pending_results"]),
    }
    return log_extra


@g.my_app.callback("pop_async_inference_results")
def pop_async_inference_results(api: sly.Api, task_id, context, state, app_logger):
    inference_request_uuid = state.get("inference_request_uuid", None)
    if inference_request_uuid is None:
        raise ValueError("Error: 'inference_request_uuid' is required.")
    if inference_request_uuid not in g.inference_requests:
        raise ValueError(f"Inference request with uuid '{inference_request_uuid}' not found")

    # Copy results
    inference_request = g.inference_requests[inference_request_uuid].copy()
    inference_request["pending_results"] = inference_request["pending_results"].copy()

    # Clear the queue `pending_results`
    g.inference_requests[inference_request_uuid]["pending_results"].clear()

    inference_request["progress"] = _convert_sly_progress_to_dict(
        inference_request["progress"]
    )

    # Logging
    log_extra = _get_log_extra_for_inference_request(
        inference_request_uuid, inference_request
    )
    app_logger.debug("Sending inference delta results with uuid:", extra=log_extra)
    g.my_app.send_response(context["request_id"], data=inference_request)


@g.my_app.callback("get_inference_result")
def get_inference_result(api: sly.Api, task_id, context, state, app_logger):
    inference_request_uuid = state.get("inference_request_uuid", None)
    if inference_request_uuid is None:
        raise ValueError("Error: 'inference_request_uuid' is required.")
    if inference_request_uuid not in g.inference_requests:
        raise ValueError(f"Inference request with uuid '{inference_request_uuid}' not found")

    inference_request = g.inference_requests[inference_request_uuid].copy()

    inference_request["progress"] = _convert_sly_progress_to_dict(
        inference_request["progress"]
    )

    # Logging
    log_extra = _get_log_extra_for_inference_request(
        inference_request_uuid, inference_request
    )
    app_logger.debug(
        "Sending inference result with uuid:",
        extra=log_extra,
    )

    g.my_app.send_response(context["request_id"], data=inference_request)


@g.my_app.callback("stop_inference")
def stop_inference(api: sly.Api, task_id, context, state, app_logger):
    inference_request_uuid = state.get("inference_request_uuid")
    if inference_request_uuid is None:
        raise ValueError("Error: 'inference_request_uuid' is required.")
    if inference_request_uuid not in g.inference_requests:
        raise ValueError(f"Inference request with uuid '{inference_request_uuid}' not found")
    inference_request = g.inference_requests[inference_request_uuid]
    inference_request["cancel_inference"] = True
    g.my_app.send_response(context["request_id"], data={"message": "Inference will be stopped.", "success": True})


@g.my_app.callback("clear_inference_request")
def clear_inference_request(api: sly.Api, task_id, context, state, app_logger):
    inference_request_uuid = state.get("inference_request_uuid")
    if inference_request_uuid is None:
        raise ValueError("Error: 'inference_request_uuid' is required.")
    if inference_request_uuid not in g.inference_requests:
        raise ValueError(f"Inference request with uuid '{inference_request_uuid}' not found")
    
    del g.inference_requests[inference_request_uuid]
    app_logger.debug("Removed an inference request:", extra={"uuid": inference_request_uuid})
    g.my_app.send_response(context["request_id"], data={"message": "Inference request removed.", "success": True})


@g.my_app.callback("get_inference_progress")
def get_inference_progress(api: sly.Api, task_id, context, state, app_logger):
    inference_request_uuid = state.get("inference_request_uuid")
    if inference_request_uuid is None:
        raise ValueError("Error: 'inference_request_uuid' is required.")
    if inference_request_uuid not in g.inference_requests:
        raise ValueError(f"Inference request with uuid '{inference_request_uuid}' not found")

    inference_request = g.inference_requests[inference_request_uuid].copy()
    inference_request["progress"] = _convert_sly_progress_to_dict(
        inference_request["progress"]
    )

    # Logging
    log_extra = _get_log_extra_for_inference_request(
        inference_request_uuid, inference_request
    )
    app_logger.debug(
        "Sending inference progress with uuid:",
        extra=log_extra,
    )

    # Ger rid of `pending_results` to less response size
    inference_request["pending_results"] = []
    g.my_app.send_response(context["request_id"], data=inference_request)

# def debug_inference1():
#     image_id = 927270
#     image_path = f"./data/images/{image_id}.jpg"
#     if not sly.fs.file_exists(image_path):
#         g.my_app.public_api.image.download_path(image_id, image_path)
#     res = nn_utils.perform_inference(g.model, image_path, topn=5)
# #
# #
# def debug_inference2():
#     image_id = 927270
#     img_np = cv2.cvtColor(g.my_app.public_api.image.download_np(image_id), cv2.COLOR_BGR2RGB)
#     res = nn_utils.perform_inference(g.model, img_np, topn=5)
# #
# #
# def debug_inference3():
#     image_id = 927270
#     img_np = cv2.cvtColor(g.my_app.public_api.image.download_np(image_id), cv2.COLOR_BGR2RGB)
#     res = nn_utils.perform_inference_batch(g.model, [img_np, img_np, img_np], topn=5)
#

def main():
    sly.logger.info("Script arguments", extra={
        "context.teamId": g.team_id,
        "context.workspaceId": g.workspace_id,
        "modal.state.slyFile": g.remote_weights_path,
        "device": g.device
    })

    nn_utils.download_model_and_configs()
    nn_utils.construct_model_meta()
    nn_utils.deploy_model()
    w.workflow_input(g.api, g.remote_weights_path)
    # debug_inference1()
    # debug_inference2()
    # debug_inference3()

    sly.logger.info("nps will be converted to RGB")
    g.my_app.run()


# @TODO: readme + gif - how to replace tag2urls file + release another app
if __name__ == "__main__":
    sly.main_wrapper("main", main)
