import copy
import os

import numpy as np
import sly_globals as g

import supervisely as sly
from mmpretrain.datasets.base_dataset import BaseDataset
from mmpretrain.datasets.builder import DATASETS
from mmpretrain.datasets.multi_label import MultiLabelDataset


@DATASETS.register_module()
class SuperviselySingleLabel(BaseDataset):

    CLASSES = None

    def __init__(self, project_dir, data_prefix, pipeline, test_mode=False):
        self.gt_labels = sly.json.load_json_file(os.path.join(project_dir, "gt_labels.json"))
        SuperviselySingleLabel.CLASSES = sorted(self.gt_labels, key=self.gt_labels.get)
        self.split_name = data_prefix
        self.items = sly.json.load_json_file(os.path.join(project_dir, "splits.json"))[
            self.split_name
        ]
        self.project_fs = sly.Project(project_dir, sly.OpenMode.READ)
        mm_ann = self.create_mm_ann()
        mm_ann_path = os.path.join(project_dir, f"{self.split_name}_mm_ann.json")
        sly.json.dump_json_file(mm_ann, mm_ann_path)

        super(SuperviselySingleLabel, self).__init__(
            ann_file=mm_ann_path,
            data_prefix=self.split_name,
            pipeline=pipeline,
            test_mode=test_mode,
        )

    def create_mm_ann(self):
        classes_set = set(SuperviselySingleLabel.CLASSES)
        data_list = []
        for paths in self.items:
            img_path = os.path.join(g.root_source_dir, paths["img_path"])
            ann_path = os.path.join(g.root_source_dir, paths["ann_path"])
            if not sly.fs.file_exists(img_path):
                sly.logger.warning(f"File {img_path} not found and item will be skipped")
                continue
            if not sly.fs.file_exists(ann_path):
                sly.logger.warning(f"File {ann_path} not found and item will be skipped")
                continue

            ann = sly.Annotation.load_json_file(ann_path, self.project_fs.meta)
            img_tags = {tag.name for tag in ann.img_tags}
            valid_tags = list(img_tags.intersection(classes_set))

            if len(valid_tags) != 1:
                sly.logger.warning(
                    f"File {ann_path} has {len(valid_tags)} gt labels, expected exactly 1"
                )
                continue
            gt_label = valid_tags[0]
            gt_index = self.gt_labels[gt_label]
            data_list.append({"img_path": img_path, "gt_label": gt_index})

        mm_annotations = {"metainfo": {"classes": SuperviselySingleLabel.CLASSES}, "data_list": data_list}
        return mm_annotations


@DATASETS.register_module()
class SuperviselyMultiLabel(MultiLabelDataset):

    CLASSES = None

    def __init__(self, project_dir, data_prefix, pipeline, test_mode=False):
        self.gt_labels = sly.json.load_json_file(os.path.join(project_dir, "gt_labels.json"))
        SuperviselyMultiLabel.CLASSES = sorted(self.gt_labels, key=self.gt_labels.get)
        self.split_name = data_prefix
        self.items = sly.json.load_json_file(os.path.join(project_dir, "splits.json"))[
            self.split_name
        ]
        self.project_fs = sly.Project(project_dir, sly.OpenMode.READ)
        mm_ann = self.create_mm_ann()
        mm_ann_path = os.path.join(project_dir, f"{self.split_name}_mm_ann.json")
        sly.json.dump_json_file(mm_ann, mm_ann_path)
        super(SuperviselyMultiLabel, self).__init__(
            ann_file=mm_ann_path,
            data_prefix=self.split_name,
            pipeline=pipeline,
            test_mode=test_mode,
        )

    def create_mm_ann(self):
        classes_set = set(SuperviselyMultiLabel.CLASSES)
        data_list = []
        for paths in self.items:
            img_path = os.path.join(g.root_source_dir, paths["img_path"])
            ann_path = os.path.join(g.root_source_dir, paths["ann_path"])
            if not sly.fs.file_exists(img_path):
                sly.logger.warning(f"File {img_path} not found and item will be skipped")
                continue
            if not sly.fs.file_exists(ann_path):
                sly.logger.warning(f"File {ann_path} not found and item will be skipped")
                continue

            ann = sly.Annotation.load_json_file(ann_path, self.project_fs.meta)
            img_tags = {tag.name for tag in ann.img_tags}
            gt_labels = []
            for tag in img_tags:
                if tag in classes_set:
                    gt_labels.append(self.gt_labels[tag])

            if not gt_labels:
                sly.logger.warning(f"File {ann_path} has no valid tags")
                continue

            data_list.append({"img_path": img_path, "gt_label": gt_labels})
        mm_annotations = {"metainfo": {"classes": SuperviselyMultiLabel.CLASSES}, "data_list": data_list}
        return mm_annotations
