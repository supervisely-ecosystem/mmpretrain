import cv2
import numpy as np
import supervisely as sly

from supervisely.sly_logger import logger

import globals as g


def get_images_ids_to_indexes_mapping(images_ids):
    imagesids2indexes = {}

    for index, image_id in enumerate(images_ids):
        imagesids2indexes.setdefault(image_id, []).append(index)

    return imagesids2indexes


def get_nps_images(images_ids):
    uniqueids2indexes = get_images_ids_to_indexes_mapping(images_ids)

    unique_images_ids = list(uniqueids2indexes.keys())

    images_infos = []
    for image_id in unique_images_ids:
        images_infos.append(g.api.image.get_info_by_id(image_id))
    images_ids = np.asarray(images_ids)

    dataset2ids = {}
    for index, image_info in enumerate(images_infos):  # group images by datasets
        dataset2ids.setdefault(image_info.dataset_id, []).append(image_info.id)

    images_nps = [_ for _ in range(len(images_ids))]  # back to plain

    for ds_id, ids_batch in dataset2ids.items():
        nps_for_ds = g.api.image.download_nps(dataset_id=ds_id, ids=ids_batch)

        for index, image_id in enumerate(ids_batch):
            for image_index in uniqueids2indexes[image_id]:
                images_nps[image_index] = cv2.cvtColor(nps_for_ds[index], cv2.COLOR_BGR2RGB)

    return np.asarray(images_nps)


def crop_images(images_nps, rectangles, padding=0):
    if rectangles is None:
        return images_nps

    elif len(rectangles) != len(images_nps):
        logger.error(f'{len(rectangles)=} != {len(images_nps)=}')
        raise ValueError(f'{len(rectangles)=} != {len(images_nps)=}')

    cropped_images = []
    for img_np, rectangle in zip(images_nps, rectangles):
        try:
            top, left, bottom, right = get_bbox_with_padding(rectangle=rectangle, pad_percent=padding,
                                                             img_size=img_np.shape[:2])

            rect = sly.Rectangle(top, left, bottom, right)
            cropping_rect = rect.crop(sly.Rectangle.from_size(img_np.shape[:2]))[0]
            cropped_image = sly.image.crop(img_np, cropping_rect)
            cropped_images.append(cropped_image)
        except Exception as ex:
            cropped_images.append(None)
            logger.warning(f'Cannot crop image: {ex}')

    return np.asarray(cropped_images)


def get_bbox_with_padding(rectangle, pad_percent, img_size):
    top, left, bottom, right = rectangle
    height, width = img_size

    if pad_percent > 0:
        sly.logger.debug("before padding", extra={"top": top, "left": left, "right": right, "bottom": bottom})
        pad_lr = int((right - left) / 100 * pad_percent)
        pad_ud = int((bottom - top) / 100 * pad_percent)
        top = max(0, top - pad_ud)
        bottom = min(height - 1, bottom + pad_ud)
        left = max(0, left - pad_lr)
        right = min(width - 1, right + pad_lr)
        sly.logger.debug("after padding", extra={"top": top, "left": left, "right": right, "bottom": bottom})

    return [top, left, bottom, right]
