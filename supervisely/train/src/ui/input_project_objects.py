import os
from datetime import datetime, timezone
import random
from collections import namedtuple
import supervisely as sly
import sly_globals as g
from sly_train_progress import get_progress_cb, reset_progress
from supervisely.io.fs import get_file_ext, get_file_name, mkdir
from supervisely.io.json import dump_json_file, load_json_file


progress_index = 1
_images_infos = None  # dataset_name -> image_name -> image_info
_cache_base_filename = os.path.join(g.my_app.data_dir, "images_info")
_cache_path = f"{_cache_base_filename}.db"
project_fs: sly.Project = None
_image_id_to_paths = {}
CNT_GRID_COLUMNS = 3


def prepare_ui_classes(project_meta):
    ui_classes = []
    classes_selected = []
    for obj_class in project_meta.obj_classes:
        if obj_class.geometry_type in [
            sly.Point,
            sly.PointLocation,
            sly.Cuboid,
        ]:
            continue
        obj_class: sly.ObjClass
        ui_classes.append(obj_class.to_json())
        classes_selected.append(True)
    return ui_classes, classes_selected, [False] * len(classes_selected)


ui_classes, classes_selected, classes_disabled = prepare_ui_classes(g.project_meta)


def init(data, state):
    data["classes"] = ui_classes
    state["classesSelected"] = classes_selected
    state["classesDisabled"] = classes_disabled

    state["cropPadding"] = 0
    state["autoSize"] = True
    state["inputWidth"] = 256
    state["inputHeight"] = 256
    state["showPreviewProgress"] = False

    # preview
    data["progress"] = 0
    data["started"] = False
    data["previewProgress"] = 0
    data["showEmptyMessage"] = False
    data["finished"] = False
    data["preview"] = {
        "content": {},
        "options": {
            "opacity": 0.5,
            "fillRectangle": False,
            "enableZoom": False,
            "syncViews": False,
        },
    }

def restart(data, state):
    data["preview"] = {
        "content": {},
        "options": {
            "opacity": 0.5,
            "fillRectangle": False,
            "enableZoom": False,
            "syncViews": False,
        },
    }
    data["previewProgress"] = 0
    state["showPreviewProgress"] = False

def get_selected_classes_from_ui(selected_classes):
    ui_classes = g.api.task.get_field(g.task_id, "data.classes")
    return [
        obj_class["name"]
        for obj_class, is_selected in zip(ui_classes, selected_classes)
        if is_selected
    ]


def resize_crop(img, ann, out_size):
    img = sly.image.resize(img, out_size)
    ann = ann.resize(out_size)
    return img, ann


def unpack_single_crop(crop, image_name):
    crop = crop[0][image_name]
    flat_crops = []
    for sublist in crop:
        for crop in sublist:
            flat_crops.append(crop)

    return flat_crops


def unpack_crops(crops, original_names):
    img_nps = []
    anns = []
    img_names = []
    name_idx = 0
    for crop, original_name in zip(crops, original_names):
        for label_crop in crop[original_name]:
            for img_np, ann in label_crop:
                img_nps.append(img_np)
                anns.append(ann)
                for label in ann.labels:
                    name_idx += 1
                    img_names.append(
                        f"{get_file_name(original_name)}_{label.obj_class.name}_{name_idx}_{label.obj_class.sly_id}.png"
                    )

    return img_nps, anns, img_names


def crop_and_resize_objects(img_nps, anns, app_state, selected_classes, original_names):
    crops = []
    crop_padding = {
        "top": f'{app_state["cropPadding"]}%',
        "left": f'{app_state["cropPadding"]}%',
        "right": f'{app_state["cropPadding"]}%',
        "bottom": f'{app_state["cropPadding"]}%',
    }

    for img_np, ann, original_name in zip(img_nps, anns, original_names):
        img_dict = {original_name: []}
        if len(ann.labels) == 0:
            crops.append(img_dict)
            continue

        for class_name in selected_classes:
            objects_crop = sly.aug.instance_crop(img_np, ann, class_name, False, crop_padding)
            if app_state["autoSize"] is False:
                resized_crop = []
                for crop_img, crop_ann in objects_crop:
                    crop_img, crop_ann = resize_crop(
                        crop_img,
                        crop_ann,
                        (app_state["inputHeight"], app_state["inputWidth"]),
                    )
                    resized_crop.append((crop_img, crop_ann))
                img_dict[original_name].append(resized_crop)
            else:
                img_dict[original_name].append(objects_crop)

        crops.append(img_dict)
    return crops


def write_images(crop_nps, crop_names, img_dir):
    for crop_np, crop_name in zip(crop_nps, crop_names):
        img_path = os.path.join(img_dir, crop_name)
        sly.image.write(img_path, crop_np)


def dump_anns(crop_anns, crop_names, ann_dir):
    for crop_ann, crop_name in zip(crop_anns, crop_names):
        ann_path = f"{os.path.join(ann_dir, crop_name)}.json"
        ann_json = crop_ann.to_json()
        dump_json_file(ann_json, ann_path)


def create_img_infos(project_fs):
    tag_id_map = {tag["name"]: tag["id"] for tag in project_fs.meta.tag_metas.to_json()}
    images_infos = []
    for dataset_fs in project_fs:
        img_info_dir = os.path.join(dataset_fs.directory, "img_info")
        mkdir(img_info_dir)
        for idx, item_name in enumerate(os.listdir(dataset_fs.item_dir)):
            item_ext = get_file_ext(item_name).lstrip(".")
            item_path = os.path.join(dataset_fs.item_dir, item_name)
            item = sly.image.read(item_path)
            h, w = item.shape[:2]
            item_size = os.path.getsize(item_path)
            created_at = datetime.fromtimestamp(
                os.stat(item_path).st_ctime, tz=timezone.utc
            ).strftime("%d-%m-%Y %H:%M:%S")
            modified_at = datetime.fromtimestamp(
                os.stat(item_path).st_mtime, tz=timezone.utc
            ).strftime("%d-%m-%Y %H:%M:%S")

            item_ann_path = os.path.join(dataset_fs.ann_dir, f"{item_name}.json")
            ann_json = load_json_file(item_ann_path)
            ann = sly.Annotation.from_json(ann_json, project_fs.meta)
            tags = ann.img_tags
            tags_json = tags.to_json()
            labels_count = len(ann.labels)

            tags_img_info = []
            for tag in tags_json:
                tag_info = {
                    "entityId": None,
                    "tagId": tag_id_map[tag["name"]],
                    "id": None,
                    "labelerLogin": tag["labelerLogin"],
                    "createdAt": tag["createdAt"],
                    "updatedAt": tag["updatedAt"],
                    "name": tag["name"],
                }
                tags_img_info.append(tag_info)

            item_img_info = {
                "id": idx,
                "name": item_name,
                "link": "",
                "hash": "",
                "mime": f"image/{item_ext}",
                "ext": item_ext,
                "size": item_size,
                "width": w,
                "height": h,
                "labels_count": labels_count,
                "dataset_id": dataset_fs.name,
                "created_at": created_at,
                "updated_at": modified_at,
                "meta": {},
                "path_original": "",
                "full_storage_url": "",
                "tags": tags_img_info,
            }
            save_path = os.path.join(img_info_dir, f"{item_name}.json")
            dump_json_file(item_img_info, save_path)
            images_infos.append(item_img_info)
    return images_infos


def convert_object_tags(project_meta):
    for tag_meta in project_meta.tag_metas:
        if tag_meta.applicable_to == sly.TagApplicableTo.OBJECTS_ONLY:
            new_tag_meta = tag_meta.clone(applicable_to=sly.TagApplicableTo.ALL)
            project_meta = project_meta.delete_tag_meta(tag_meta.name)
            project_meta = project_meta.add_tag_meta(new_tag_meta)

    return project_meta


def copy_tags(crop_anns):
    new_anns = []
    for ann in crop_anns:
        for label in ann.labels:
            label_tags = label.tags
            new_ann = ann.add_tags(label_tags)
            new_anns.append(new_ann)
    return new_anns


@g.my_app.callback("download_project_objects")
@sly.timeit
@g.my_app.ignore_errors_and_show_dialog_window()
def download_project_objects(api: sly.Api, task_id, context, state, app_logger):
    try:
        mkdir(g.project_dir, remove_content_if_exists=True)
        project_meta_path = os.path.join(g.project_dir, "meta.json")
        g.project_meta = convert_object_tags(g.project_meta)
        project_meta_json = g.project_meta.to_json()
        dump_json_file(project_meta_json, project_meta_path)
        datasets = api.dataset.get_list(g.project_id, recursive=True)
        for dataset in datasets:
            ds_dir = os.path.join(g.project_dir, dataset.name)
            img_dir = os.path.join(ds_dir, "img")
            ann_dir = os.path.join(ds_dir, "ann")

            mkdir(ds_dir)
            mkdir(img_dir)
            mkdir(ann_dir)
            images_infos = api.image.get_list(dataset.id)
            download_progress = get_progress_cb(
                progress_index, "Download project", g.project_info.items_count * 2
            )
            for batch in sly.batched(images_infos):
                image_ids = [image_info.id for image_info in batch]
                image_names = [image_info.name for image_info in batch]
                ann_infos = api.annotation.download_batch(
                    dataset.id, image_ids, progress_cb=download_progress
                )

                image_nps = api.image.download_nps(
                    dataset.id, image_ids, progress_cb=download_progress
                )
                anns = [
                    sly.Annotation.from_json(ann_info.annotation, g.project_meta)
                    for ann_info in ann_infos
                ]
                selected_classes = get_selected_classes_from_ui(state["classesSelected"])
                crops = crop_and_resize_objects(
                    image_nps, anns, state, selected_classes, image_names
                )
                crop_nps, crop_anns, crop_names = unpack_crops(crops, image_names)
                crop_anns = copy_tags(crop_anns)
                write_images(crop_nps, crop_names, img_dir)
                dump_anns(crop_anns, crop_names, ann_dir)

        reset_progress(progress_index)

        global project_fs
        project_fs = sly.Project(g.project_dir, sly.OpenMode.READ)
        g.images_infos = create_img_infos(project_fs)
    except Exception as e:
        reset_progress(progress_index)
        raise e

    # items_count = g.project_stats["objects"]["total"]["objectsInDataset"]
    items_count = project_fs.total_items
    train_percent = 80
    train_count = int(items_count / 100 * train_percent)
    random_split = {
        "count": {"total": items_count, "train": train_count, "val": items_count - train_count},
        "percent": {"total": 100, "train": train_percent, "val": 100 - train_percent},
        "shareImagesBetweenSplits": False,
        "sliderDisabled": False,
    }

    fields = [
        {"field": "state.allowTestartStep1", "payload": True},
        {"field": "state.restartFrom", "payload": None},
        {"field": "data.done1", "payload": True},
        {"field": "state.collapsed2", "payload": False},
        {"field": "state.disabled2", "payload": False},
        {"field": "state.activeStep", "payload": 2},
        {"field": "state.totalImagesCount", "payload": items_count},
        {"field": "state.randomSplit", "payload": random_split},
    ]
    g.api.app.set_fields(g.task_id, fields)


@sly.timeit
def upload_preview(crops):
    if len(crops) == 0:
        g.api.task.set_fields(g.task_id, [{"field": "data.showEmptyMessage", "payload": True}])
        return

    upload_src_paths = []
    upload_dst_paths = []
    for idx, (cur_img, cur_ann) in enumerate(crops):
        img_name = "{:03d}.png".format(idx)
        remote_path = "/temp/{}/{}".format(g.task_id, img_name)
        if g.api.file.exists(g.team_id, remote_path):
            g.api.file.remove(g.team_id, remote_path)
        local_path = "{}/{}".format(g.my_app.data_dir, img_name)
        sly.image.write(local_path, cur_img)
        upload_src_paths.append(local_path)
        upload_dst_paths.append(remote_path)

    g.api.file.remove(g.team_id, "/temp/{}/".format(g.task_id))

    def _progress_callback(monitor):
        if hasattr(monitor, "last_percent") is False:
            monitor.last_percent = 0
        cur_percent = int(monitor.bytes_read * 100.0 / monitor.len)
        if cur_percent - monitor.last_percent > 15 or cur_percent == 100:
            g.api.task.set_fields(
                g.task_id, [{"field": "data.previewProgress", "payload": cur_percent}]
            )
            monitor.last_percent = cur_percent

    upload_results = g.api.file.upload_bulk(
        g.team_id, upload_src_paths, upload_dst_paths, _progress_callback
    )
    # clean local data
    for local_path in upload_src_paths:
        sly.fs.silent_remove(local_path)
    return upload_results


@g.my_app.callback("preview_objects")
@sly.timeit
def preview_objects(api: sly.Api, task_id, context, state, app_logger):
    api.task.set_fields(
        task_id,
        [
            {"field": "data.previewProgress", "payload": 0},
            {"field": "state.showPreviewProgress", "payload": True},
        ],
    )
    image_id = random.choice(g.image_ids)
    image_info = api.image.get_info_by_id(image_id)
    image_name = image_info.name

    img = api.image.download_np(image_info.id)
    ann_json = api.annotation.download(image_id).annotation
    ann = sly.Annotation.from_json(ann_json, g.project_meta)

    selected_classes = get_selected_classes_from_ui(state["classesSelected"])
    single_crop = crop_and_resize_objects([img], [ann], state, selected_classes, [image_name])
    single_crop = unpack_single_crop(single_crop, image_name)
    single_crop = [(img, ann)] + single_crop

    grid_data = {}
    grid_layout = [[] for _ in range(CNT_GRID_COLUMNS)]

    upload_results = upload_preview(single_crop)
    for idx, info in enumerate(upload_results):
        if idx > 8:
            break

        if idx == 0:
            grid_data[idx] = {
                "url": info.storage_path,
                "image_name": f"Original image ({image_name})",
                "figures": [label.to_json() for label in single_crop[idx][1].labels],
            }
        else:
            object_tags_names = []
            for tag in single_crop[idx][1].labels[0].tags.to_json():
                if tag.get("value") is None:
                    object_tags_names.append(tag.get("name"))
                else:
                    object_tags_names.append(f"{tag.get('name')}: {tag.get('value')}")

            grid_data[idx] = {
                "url": info.storage_path,
                "tag_names": object_tags_names,
                "figures": [],
            }
        grid_layout[idx % CNT_GRID_COLUMNS].append(idx)

    fields = []
    if grid_data:
        content = {
            "projectMeta": g.project_meta_json,
            "annotations": grid_data,
            "layout": grid_layout,
        }
        fields.append({"field": "data.preview.content", "payload": content})
    fields.append({"field": "state.showPreviewProgress", "payload": False})
    api.task.set_fields(task_id, fields)


def get_image_info_from_cache(dataset_name, item_name):
    global project_fs
    dataset_fs = project_fs.datasets.get(dataset_name)
    img_info_path = dataset_fs.get_img_info_path(item_name)
    image_info_dict = sly.json.load_json_file(img_info_path)
    ImageInfo = namedtuple("ImageInfo", image_info_dict)
    info = ImageInfo(**image_info_dict)

    # add additional info - helps to save split paths to txt files
    _image_id_to_paths[info.id] = dataset_fs.get_item_paths(item_name)._asdict()

    return info


def get_paths_by_image_id(image_id):
    return _image_id_to_paths[image_id]
