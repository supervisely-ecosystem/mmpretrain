from collections import defaultdict
import os
import supervisely as sly
from supervisely.io.fs import mkdir
import sly_globals as g
import input_project
import input_project_objects
import random
import tags

report = []
final_tags = []
final_tags2images = defaultdict(lambda: defaultdict(list))

remote_images_dir = os.path.join("temp", str(g.task_id))
artifacts_example_img_dir = os.path.join(g.my_app.data_dir, "artifacts", "example_images")


def init(data, state):
    data["done4"] = False
    state["collapsed4"] = True
    state["disabled4"] = True
    data["validationReport"] = None
    data["cntErrors"] = 0
    data["cntWarnings"] = 0
    data["report"] = None
    state["final_train_size"] = 0

    state["isValidating"] = False


@g.my_app.callback("validate_data")
@sly.timeit
@g.my_app.ignore_errors_and_show_dialog_window()
def validate_data(api: sly.Api, task_id, context, state, app_logger):
    # g.api.app.set_field(g.task_id, "state.isValidating", True)
    report.clear()
    final_tags.clear()
    final_tags2images.clear()

    if state["trainData"] == "objects":
        mkdir(artifacts_example_img_dir, True)

    report.append({
        "type": "info",
        "title": "Total tags in project",
        "count": len(g.project_meta.tag_metas),
        "description": None
    })

    report.append({
        "title": "Tags unavailable for training",
        "count": len(tags.disabled_tags),
        "type": "info",
        "description": "See previous step for more info"
    })

    selected_tags = tags.selected_tags  # state["selectedTags"]
    report.append({
        "title": "Selected tags for training",
        "count": len(selected_tags),
        "type": "info",
        "description": None
    })

    report.append({
        "type": "info",
        "title": "Total images in project",
        "count": g.project_info.items_count,
    })

    report.append({
        "title": "Images without tags",
        "count": len(tags.images_without_tags),
        "type": "warning" if len(tags.images_without_tags) > 0 else "pass",
        "description": "Such images don't have any tags so they will ignored and will not be used for training. "
    })

    images_before_validation = []
    num_images_before_validation = 0
    for tag_name in selected_tags:
        for split, infos in tags.tag2images[tag_name].items():
            for info in infos:
                if info.id not in images_before_validation:
                    images_before_validation.append(info.id)
    num_images_before_validation = len(images_before_validation)
    report.append({
        "title": "Images with training tags",
        "count": num_images_before_validation,
        "type": "error" if num_images_before_validation == 0 else "pass",
        "description": "Images that have one of the selected tags assigned (before validation)"
    })

    collisions = defaultdict(lambda: defaultdict(int))
    for tag_name in selected_tags:
        for split, infos in tags.tag2images[tag_name].items():
            for info in infos:
                collisions[split][info.id] += 1
    
    if state["cls_mode"] == "one_label":
        num_collision_images = 0
        for split, split_collisions in collisions.items():
            for image_id, counter in split_collisions.items():
                if counter > 1:
                    num_collision_images += 1
        report.append({
            "title": "Images with tags collisions",
            "count": num_collision_images,
            "type": "warning" if num_collision_images > 0 else "pass",
            "description": "Images with more than one training tags assigned, they will be removed from train/val sets. Use app 'Tags Co-Occurrence Matrix' to discover such images"
        })

    # remove collision images from sets if cls_mode == 'one_label'
    for tag_name in selected_tags:
        for split, infos in tags.tag2images[tag_name].items():
            _final_infos = []
            for info in infos:
                if collisions[split][info.id] != 1 and state["cls_mode"] == "one_label":
                    continue
                _final_infos.append(info)
            if len(_final_infos) > 0:
                final_tags2images[tag_name][split].extend(_final_infos)
    for tag_name in selected_tags:
        if tag_name in final_tags2images and len(final_tags2images[tag_name]["train"]) > 0:
            final_tags.append(tag_name)

    tags_examples = defaultdict(list)
    for tag_name, infos in final_tags2images.items():
        for info in (infos['train'] + infos['val'])[:tags._max_examples_count]:
            if state["trainData"] == "objects":
                info = upload_img_example_to_files(api, info)
                tags_examples[tag_name].append(
                    info.full_storage_url
                )
            else:
                tags_examples[tag_name].append(
                    info.full_storage_url
                )
    sly.json.dump_json_file(tags_examples, os.path.join(g.info_dir, "tag2urls.json"))

    # save splits
    split_paths = {}
    final_images_count = 0
    final_train_size = 0
    final_val_size = 0
    for tag_name, splits in final_tags2images.items():
        for split_name, infos in splits.items():
            if split_name not in split_paths.keys():
                split_paths[split_name] = {}
            for info in infos:
                if state["trainData"] == "images":
                    paths = input_project.get_paths_by_image_id(info.id)
                else:
                    paths = input_project_objects.get_paths_by_image_id(info.id)
                if info.id not in split_paths[split_name].keys(): # without duplicates
                    split_paths[split_name][info.id] = paths
    final_split_paths = {}
    for split_name, split in split_paths.items():
        items_to_add = list(split.values())
        final_split_paths[split_name] = items_to_add
        if split_name == "train":
            final_train_size += len(items_to_add)
        elif split_name == "val":
            final_val_size += len(items_to_add)
    final_images_count = final_train_size + final_val_size

    report.append({
        "title": "Final images count",
        "count": final_images_count,
        "type": "error" if final_images_count == 0 else "pass",
        "description": "Number of images (train + val) after collisions removal"
    })
    report.append({
        "title": "Train set size",
        "count": final_train_size,
        "type": "error" if final_train_size == 0 else "pass",
        "description": "Size of training set after collisions removal"
    })
    report.append({
        "title": "Val set size",
        "count": final_val_size,
        "type": "error" if final_val_size == 0 else "pass",
        "description": "Size of validation set after collisions removal"
    })

    type = "pass"
    if len(final_tags) < 2:
        type = "error"
    elif len(final_tags) != len(selected_tags):
        type = "error"
    report.append({
        "title": "Final training tags",
        "count": len(final_tags),
        "type": type,
        "description": f"If this number differs from the number of selected tags then it means that after data "
                       f"validation and cleaning some of the selected tags "
                       f"{list(set(selected_tags) - set(final_tags))} "
                       f"have 0 examples in train set. Please restart step 3 and deselect this tags manually"
    })

    cnt_errors = 0
    cnt_warnings = 0
    for item in report:
        if item["type"] == "error":
            cnt_errors += 1
        if item["type"] == "warning":
            cnt_warnings += 1

    fields = [
        {"field": "data.report", "payload": report},
        {"field": "data.done4", "payload": True},
        {"field": "data.cntErrors", "payload": cnt_errors},
        {"field": "data.cntWarnings", "payload": cnt_warnings},
        {"field": "state.final_train_size", "payload": final_train_size}
    ]
    if cnt_errors == 0:
        # save selected tags
        gt_labels = {tag_name: idx for idx, tag_name in enumerate(final_tags)}
        sly.json.dump_json_file(gt_labels, os.path.join(g.project_dir, "gt_labels.json"))
        sly.json.dump_json_file(gt_labels, os.path.join(g.info_dir, "gt_labels.json"))
        sly.json.dump_json_file(final_split_paths, os.path.join(g.project_dir, "splits.json"))

        fields.extend([
            {"field": "state.collapsed5", "payload": False},
            {"field": "state.disabled5", "payload": False},
            {"field": "state.activeStep", "payload": 5},
            {"field": "state.isValidating", "payload": False},
        ])

    
    g.api.app.set_fields(g.task_id, fields)


def get_random_image():
    rand_key = random.choice(list(final_tags2images.keys()))
    info = random.choice(final_tags2images[rand_key]['train'])
    # ImageInfo = namedtuple('ImageInfo', image_info_dict)
    # info = ImageInfo(**image_info_dict)
    return info


def upload_img_example_to_files(api, info):
    img_path = os.path.join(g.project_dir, info.dataset_id, "img", info.name)
    remote_image_path = os.path.join("/mmclassification", f"{g.task_id}_{g.project_info.name}", "example_images", info.name)
    info = api.file.upload(g.team_id, img_path, remote_image_path)
    return info
