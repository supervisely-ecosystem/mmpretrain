import errno
import os
from pathlib import Path

import requests
import sly_globals as g
from sly_train_progress import get_progress_cb, init_progress, reset_progress

import supervisely as sly

local_weights_path = None


def get_models_list():
    res = [
        {
            "modelConfig": "configs/_base_/models/vgg11.py",
            "config": "configs/vgg/vgg11_8xb32_in1k.py",
            "weightsUrl": "https://download.openmmlab.com/mmclassification/v0/vgg/vgg11_batch256_imagenet_20210208-4271cd6c.pth",
            "model": "VGG-11",
            "collection": "VGG",
            "params": "132.86",
            "flops": "7.63",
            "top1": "68.75",
            "top5": "88.87",
        },
        {
            "modelConfig": "configs/_base_/models/vgg13.py",
            "config": "configs/vgg/vgg13_8xb32_in1k.py",
            "weightsUrl": "https://download.openmmlab.com/mmclassification/v0/vgg/vgg13_batch256_imagenet_20210208-4d1d6080.pth",
            "model": "VGG-13",
            "collection": "VGG",
            "params": "133.05",
            "flops": "11.34",
            "top1": "70.02",
            "top5": "89.46",
        },
        {
            "modelConfig": "configs/_base_/models/vgg16.py",
            "config": "configs/vgg/vgg16_8xb32_in1k.py",
            "weightsUrl": "https://download.openmmlab.com/mmclassification/v0/vgg/vgg16_batch256_imagenet_20210208-db26f1a5.pth",
            "model": "VGG-16",
            "collection": "VGG",
            "params": "138.36",
            "flops": "15.5",
            "top1": "71.62",
            "top5": "90.49",
        },
        {
            "modelConfig": "configs/_base_/models/vgg19.py",
            "config": "configs/vgg/vgg19_8xb32_in1k.py",
            "weightsUrl": "https://download.openmmlab.com/mmclassification/v0/vgg/vgg19_bn_batch256_imagenet_20210208-da620c4f.pth",
            "model": "VGG-19",
            "collection": "VGG",
            "params": "143.67",
            "flops": "19.67",
            "top1": "72.41",
            "top5": "90.80",
        },
        {
            "modelConfig": "configs/_base_/models/vgg11bn.py",
            "config": "configs/vgg/vgg11bn_8xb32_in1k.py",
            "weightsUrl": "https://download.openmmlab.com/mmclassification/v0/vgg/vgg11_bn_batch256_imagenet_20210207-f244902c.pth",
            "model": "VGG-11-BN",
            "collection": "VGG",
            "params": "132.87",
            "flops": "7.64",
            "top1": "70.75",
            "top5": "90.12",
        },
        {
            "modelConfig": "configs/_base_/models/vgg13bn.py",
            "config": "configs/vgg/vgg13bn_8xb32_in1k.py",
            "weightsUrl": "https://download.openmmlab.com/mmclassification/v0/vgg/vgg13_bn_batch256_imagenet_20210207-1a8b7864.pth",
            "model": "VGG-13-BN",
            "collection": "VGG",
            "params": "133.05",
            "flops": "11.36",
            "top1": "72.15",
            "top5": "90.71",
        },
        {
            "modelConfig": "configs/_base_/models/vgg16bn.py",
            "config": "configs/vgg/vgg16bn_8xb32_in1k.py",
            "weightsUrl": "https://download.openmmlab.com/mmclassification/v0/vgg/vgg16_bn_batch256_imagenet_20210208-7e55cd29.pth",
            "model": "VGG-16-BN",
            "collection": "VGG",
            "params": "138.37",
            "flops": "15.53",
            "top1": "73.72",
            "top5": "91.68",
        },
        {
            "modelConfig": "configs/_base_/models/vgg19bn.py",
            "config": "configs/vgg/vgg19bn_8xb32_in1k.py",
            "weightsUrl": "https://download.openmmlab.com/mmclassification/v0/vgg/vgg19_bn_batch256_imagenet_20210208-da620c4f.pth",
            "model": "VGG-19-BN",
            "collection": "VGG",
            "params": "143.68",
            "flops": "19.7",
            "top1": "74.70",
            "top5": "92.24",
        },
        {
            "modelConfig": "configs/_base_/models/resnet18.py",
            "config": "configs/resnet/resnet18_8xb32_in1k.py",
            "weightsUrl": "https://download.openmmlab.com/mmclassification/v0/resnet/resnet18_batch256_imagenet_20200708-34ab8f90.pth",
            "model": "ResNet-18",
            "collection": "ResNet",
            "params": "11.69",
            "flops": "1.82",
            "top1": "70.07",
            "top5": "89.44",
        },
        {
            "modelConfig": "configs/_base_/models/resnet34.py",
            "config": "configs/resnet/resnet34_8xb32_in1k.py",
            "weightsUrl": "https://download.openmmlab.com/mmclassification/v0/resnet/resnet34_batch256_imagenet_20200708-32ffb4f7.pth",
            "model": "ResNet-34",
            "collection": "ResNet",
            "params": "21.8",
            "flops": "3.68",
            "top1": "73.85",
            "top5": "91.53",
        },
        {
            "modelConfig": "configs/_base_/models/resnet50.py",
            "config": "configs/resnet/resnet50_8xb32_in1k.py",
            "weightsUrl": "https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_batch256_imagenet_20200708-cfb998bf.pth",
            "model": "ResNet-50",
            "collection": "ResNet",
            "params": "25.56",
            "flops": "4.12",
            "top1": "76.55",
            "top5": "93.15",
        },
        {
            "modelConfig": "configs/_base_/models/resnet101.py",
            "config": "configs/resnet/resnet101_8xb32_in1k.py",
            "weightsUrl": "https://download.openmmlab.com/mmclassification/v0/resnet/resnet101_batch256_imagenet_20200708-753f3608.pth",
            "model": "ResNet-101",
            "collection": "ResNet",
            "params": "44.55",
            "flops": "7.85",
            "top1": "78.18",
            "top5": "94.03",
        },
        {
            "modelConfig": "configs/_base_/models/resnet152.py",
            "config": "configs/resnet/resnet152_8xb32_in1k.py",
            "weightsUrl": "https://download.openmmlab.com/mmclassification/v0/resnet/resnet152_batch256_imagenet_20200708-ec25b1f9.pth",
            "model": "ResNet-152",
            "collection": "ResNet",
            "params": "60.19",
            "flops": "11.58",
            "top1": "78.63",
            "top5": "94.16",
        },
        {
            "modelConfig": "configs/_base_/models/resnetv1d50.py",
            "config": "configs/resnet/resnetv1d50_8xb32_in1k.py",
            "weightsUrl": "https://download.openmmlab.com/mmclassification/v0/resnet/resnetv1d50_b32x8_imagenet_20210531-db14775a.pth",
            "model": "ResNetV1D-50",
            "collection": "ResNet",
            "params": "25.58",
            "flops": "4.36",
            "top1": "77.54",
            "top5": "93.57",
        },
        {
            "modelConfig": "configs/_base_/models/resnetv1d101.py",
            "config": "configs/resnet/resnetv1d101_8xb32_in1k.py",
            "weightsUrl": "https://download.openmmlab.com/mmclassification/v0/resnet/resnetv1d101_b32x8_imagenet_20210531-6e13bcd3.pth",
            "model": "ResNetV1D-101",
            "collection": "ResNet",
            "params": "44.57",
            "flops": "8.09",
            "top1": "78.93",
            "top5": "94.48",
        },
        {
            "modelConfig": "configs/_base_/models/resnetv1d152.py",
            "config": "configs/resnet/resnetv1d152_8xb32_in1k.py",
            "weightsUrl": "https://download.openmmlab.com/mmclassification/v0/resnet/resnetv1d152_b32x8_imagenet_20210531-278cf22a.pth",
            "model": "ResNetV1D-152",
            "collection": "ResNet",
            "params": "60.21",
            "flops": "11.82",
            "top1": "79.41",
            "top5": "94.7",
        },
        {
            "modelConfig": "configs/_base_/models/resnext50_32x4d.py",
            "config": "configs/resnext/resnext50-32x4d_8xb32_in1k.py",
            "weightsUrl": "https://download.openmmlab.com/mmclassification/v0/resnext/resnext50_32x4d_b32x8_imagenet_20210429-56066e27.pth",
            "model": "ResNeXt-32x4d-50",
            "collection": "ResNext",
            "params": "25.03",
            "flops": "4.27",
            "top1": "77.90",
            "top5": "93.66",
        },
        {
            "modelConfig": "configs/_base_/models/resnext101_32x4d.py",
            "config": "configs/resnext/resnext101-32x4d_8xb32_in1k.py",
            "weightsUrl": "https://download.openmmlab.com/mmclassification/v0/resnext/resnext101_32x4d_b32x8_imagenet_20210506-e0fa3dd5.pth",
            "model": "ResNeXt-32x4d-101",
            "collection": "ResNext",
            "params": "44.18",
            "flops": "8.03",
            "top1": "78.71",
            "top5": "94.12",
        },
        {
            "modelConfig": "configs/_base_/models/resnext101_32x8d.py",
            "config": "configs/resnext/resnext101-32x8d_8xb32_in1k.py",
            "weightsUrl": "https://download.openmmlab.com/mmclassification/v0/resnext/resnext101_32x8d_b32x8_imagenet_20210506-23a247d5.pth",
            "model": "ResNeXt-32x8d-101",
            "collection": "ResNext",
            "params": "88.79",
            "flops": "16.5",
            "top1": "79.23",
            "top5": "94.58",
        },
        {
            "modelConfig": "configs/_base_/models/resnext152_32x4d.py",
            "config": "configs/resnext/resnext152-32x4d_8xb32_in1k.py",
            "weightsUrl": "https://download.openmmlab.com/mmclassification/v0/resnext/resnext152_32x4d_b32x8_imagenet_20210524-927787be.pth",
            "model": "ResNeXt-32x4d-152",
            "collection": "ResNext",
            "params": "59.95",
            "flops": "11.8",
            "top1": "78.93",
            "top5": "94.41",
        },
        {
            "modelConfig": "configs/_base_/models/seresnet50.py",
            "config": "configs/seresnet/seresnet50_8xb32_in1k.py",
            "weightsUrl": "https://download.openmmlab.com/mmclassification/v0/se-resnet/se-resnet50_batch256_imagenet_20200804-ae206104.pth",
            "model": "SE-ResNet-50",
            "params": "28.09",
            "flops": "4.13",
            "top1": "77.74",
            "top5": "93.84",
        },
        {
            "modelConfig": "configs/_base_/models/seresnet101.py",
            "config": "configs/seresnet/seresnet101_8xb32_in1k.py",
            "weightsUrl": "https://download.openmmlab.com/mmclassification/v0/se-resnet/se-resnet101_batch256_imagenet_20200804-ba5b51d4.pth",
            "model": "SE-ResNet-101",
            "collection": "SE-ResNet",
            "params": "49.33",
            "flops": "7.86",
            "top1": "78.26",
            "top5": "94.07",
        },
        {
            "modelConfig": "configs/_base_/models/shufflenet_v1_1x.py",
            "config": "configs/shufflenet_v1/shufflenet-v1-1x_16xb64_in1k.py",
            "weightsUrl": "https://download.openmmlab.com/mmclassification/v0/shufflenet_v1/shufflenet_v1_batch1024_imagenet_20200804-5d6cec73.pth",
            "model": "ShuffleNetV1 1.0x (group=3)",
            "collection": "ShuffleNet",
            "params": "1.87",
            "flops": "0.146",
            "top1": "68.13",
            "top5": "87.81",
        },
        {
            "modelConfig": "configs/_base_/models/shufflenet_v2_1x.py",
            "config": "configs/shufflenet_v2/shufflenet-v2-1x_16xb64_in1k.py",
            "weightsUrl": "https://download.openmmlab.com/mmclassification/v0/shufflenet_v2/shufflenet_v2_batch1024_imagenet_20200812-5bf4721e.pth",
            "model": "ShuffleNetV2 1.0x",
            "collection": "ShuffleNet",
            "params": "2.28",
            "flops": "0.149",
            "top1": "69.55",
            "top5": "88.92",
        },
        {
            "modelConfig": "configs/_base_/models/mobilenet_v2_1x.py",
            "config": "configs/mobilenet_v2/mobilenet-v2_8xb32_in1k.py",
            "weightsUrl": "https://download.openmmlab.com/mmclassification/v0/mobilenet_v2/mobilenet_v2_batch256_imagenet_20200708-3b2dc3af.pth",
            "model": "MobileNet V2",
            "collection": "MobileNet",
            "params": "3.5",
            "flops": "0.319",
            "top1": "71.86",
            "top5": "90.42",
        },
        {
            "modelConfig": "configs/_base_/models/mobilenet_v3/mobilenet_v3_small_imagenet.py",
            "config": "configs/mobilenet_v3/mobilenet-v3-small_8xb128_in1k.py",
            "weightsUrl": "https://download.openmmlab.com/mmclassification/v0/mobilenet_v3/mobilenet-v3-small_8xb128_in1k_20221114-bd1bfcde.pth",
            "model": "MobileNet V3 Small",
            "collection": "MobileNet",
            "params": "2.54",
            "flops": "0.06",
            "top1": "66.68",
            "top5": "86.74",
        },
        {
            "modelConfig": "configs/_base_/models/mobilenet_v3/mobilenet_v3_large_imagenet.py",
            "config": "configs/mobilenet_v3/mobilenet-v3-large_8xb128_in1k.py",
            "weightsUrl": "https://download.openmmlab.com/mmclassification/v0/mobilenet_v3/mobilenet-v3-large_8xb128_in1k_20221114-0ed9ed9a.pth",
            "model": "MobileNet V3 Large",
            "collection": "MobileNet",
            "params": "5.48",
            "flops": "0.23",
            "top1": "73.49",
            "top5": "91.31",
        },
        {
            "modelConfig": "configs/_base_/models/vit-base-p16.py",
            "config": "configs/vision_transformer/vit-base-p16_64xb64_in1k-384px.py",
            "weightsUrl": "https://download.openmmlab.com/mmclassification/v0/vit/finetune/vit-base-p16_in21k-pre-3rdparty_ft-64xb64_in1k-384_20210928-98e8652b.pth",
            "model": "ViT-B/16*",
            "collection": "ViT",
            "params": "86.86",
            "flops": "33.03",
            "top1": "84.20",
            "top5": "97.18",
        },
        {
            "modelConfig": "configs/_base_/models/vit-base-p32.py",
            "config": "configs/vision_transformer/vit-base-p32_64xb64_in1k-384px.py",
            "weightsUrl": "https://download.openmmlab.com/mmclassification/v0/vit/finetune/vit-base-p32_in21k-pre-3rdparty_ft-64xb64_in1k-384_20210928-9cea8599.pth",
            "model": "ViT-B/32*",
            "collection": "ViT",
            "params": "88.3",
            "flops": "8.56",
            "top1": "81.73",
            "top5": "96.13",
        },
        {
            "modelConfig": "configs/_base_/models/vit-large-p16.py",
            "config": "configs/vision_transformer/vit-large-p16_64xb64_in1k-384px.py",
            "weightsUrl": "https://download.openmmlab.com/mmclassification/v0/vit/finetune/vit-large-p16_in21k-pre-3rdparty_ft-64xb64_in1k-384_20210928-b20ba619.pth",
            "model": "ViT-L/16*",
            "collection": "ViT",
            "params": "304.72",
            "flops": "116.68",
            "top1": "85.08",
            "top5": "97.38",
        },
        # New
        # EfficientNetV2
        {
            "modelConfig": "configs/_base_/models/efficientnet_v2/efficientnetv2_s.py",
            "config": "configs/efficientnet_v2/efficientnetv2-s_8xb32_in21k.py",
            "weightsUrl": "https://download.openmmlab.com/mmclassification/v0/efficientnetv2/efficientnetv2-s_3rdparty_in21k_20221220-c0572b56.pth",
            "model": "EfficientNetV2 Small",
            "collection": "EfficientNet",
            "params": "21.46",
            "flops": "9.72",
            "top1": "84.29",
            "top5": "97.26",
        },
        {
            "modelConfig": "configs/_base_/models/efficientnet_v2/efficientnetv2_m.py",
            "config": "configs/efficientnet_v2/efficientnetv2-m_8xb32_in21k.py",
            "weightsUrl": "https://download.openmmlab.com/mmclassification/v0/efficientnetv2/efficientnetv2-m_3rdparty_in21k_20221220-073e944c.pth",
            "model": "EfficientNetV2 Medium",
            "collection": "EfficientNet",
            "params": "54.14",
            "flops": "26.88",
            "top1": "85.47",
            "top5": "97.76",
        },
        {
            "modelConfig": "configs/_base_/models/efficientnet_v2/efficientnetv2_l.py",
            "config": "configs/efficientnet_v2/efficientnetv2-l_8xb32_in21k.py",
            "weightsUrl": "https://download.openmmlab.com/mmclassification/v0/efficientnetv2/efficientnetv2-l_3rdparty_in21k_20221220-f28f91e1.pth",
            "model": "EfficientNetV2 Large",
            "collection": "EfficientNet",
            "params": "118.52",
            "flops": "60.14",
            "top1": "86.31",
            "top5": "97.99",
        },
        {
            "modelConfig": "configs/_base_/models/efficientnet_v2/efficientnetv2_xl.py",
            "config": "configs/efficientnet_v2/efficientnetv2-xl_8xb32_in21k.py",
            "weightsUrl": "https://download.openmmlab.com/mmclassification/v0/efficientnetv2/efficientnetv2-xl_3rdparty_in21k_20221220-b2c9329c.pth",
            "model": "EfficientNetV2 XLarge",
            "collection": "EfficientNet",
            "params": "208.12",
            "flops": "98.34",
            "top1": "86.39",
            "top5": "97.83",
        },
        # DeiT3
        {
            "modelConfig": "configs/_base_/models/deit3/deit3-small-p16-224.py",
            "config": "configs/deit3/deit3-small-p16_64xb64_in1k.py",
            "weightsUrl": "https://download.openmmlab.com/mmclassification/v0/deit3/deit3-small-p16_3rdparty_in1k_20221008-0f7c70cf.pth",
            "model": "DeiT3 Small",
            "collection": "DeiT",
            "params": "22.06",
            "flops": "4.61",
            "top1": "81.35",
            "top5": "95.31",
        },
        {
            "modelConfig": "configs/_base_/models/deit3/deit3-medium-p16-224.py",
            "config": "configs/deit3/deit3-medium-p16_64xb64_in1k.py",
            "weightsUrl": "https://download.openmmlab.com/mmclassification/v0/deit3/deit3-medium-p16_3rdparty_in1k_20221008-3b21284d.pth",
            "model": "DeiT3 Medium",
            "collection": "DeiT",
            "params": "38.85",
            "flops": "8.00",
            "top1": "82.99",
            "top5": "96.22",
        },
        {
            "modelConfig": "configs/_base_/models/deit3/deit3-base-p16-224.py",
            "config": "configs/deit3/deit3-base-p16_64xb64_in1k.py",
            "weightsUrl": "https://download.openmmlab.com/mmclassification/v0/deit3/deit3-base-p16_3rdparty_in1k_20221008-60b8c8bf.pth",
            "model": "DeiT3 Base",
            "collection": "DeiT",
            "params": "86.59",
            "flops": "17.58",
            "top1": "83.80",
            "top5": "96.55",
        },
        {
            "modelConfig": "configs/_base_/models/deit3/deit3-large-p16-224.py",
            "config": "configs/deit3/deit3-large-p16_64xb64_in1k.py",
            "weightsUrl": "https://download.openmmlab.com/mmclassification/v0/deit3/deit3-large-p16_3rdparty_in1k_20221009-03b427ea.pth",
            "model": "DeiT3 Large",
            "collection": "DeiT",
            "params": "304.37",
            "flops": "61.60",
            "top1": "84.87",
            "top5": "97.01",
        },
        {
            "modelConfig": "configs/_base_/models/deit3/deit3-huge-p14-224.py",
            "config": "configs/deit3/deit3-huge-p14_64xb32_in1k.py",
            "weightsUrl": "https://download.openmmlab.com/mmclassification/v0/deit3/deit3-huge-p14_3rdparty_in1k_20221009-e107bcb7.pth",
            "model": "DeiT3 Huge",
            "collection": "DeiT",
            "params": "632.13",
            "flops": "167.40",
            "top1": "85.21",
            "top5": "97.36",
        },
        # ConvNeXt
        {
            "modelConfig": "configs/_base_/models/convnext/convnext-tiny.py",
            "config": "configs/convnext/convnext-tiny_32xb128_in1k.py",
            "weightsUrl": "https://download.openmmlab.com/mmclassification/v0/convnext/convnext-tiny_32xb128_in1k_20221207-998cf3e9.pth",
            "model": "ConvNeXt Tiny",
            "collection": "ConvNeXt",
            "params": "28.59",
            "flops": "4.46",
            "top1": "82.14",
            "top5": "96.06",
        },
        {
            "modelConfig": "configs/_base_/models/convnext/convnext-small.py",
            "config": "configs/convnext/convnext-small_32xb128_in1k.py",
            "weightsUrl": "https://download.openmmlab.com/mmclassification/v0/convnext/convnext-small_32xb128_in1k_20221207-4ab7052c.pth",
            "model": "ConvNeXt Small",
            "collection": "ConvNeXt",
            "params": "50.22",
            "flops": "8.69",
            "top1": "83.16",
            "top5": "96.56",
        },
        {
            "modelConfig": "configs/_base_/models/convnext/convnext-base.py",
            "config": "configs/convnext/convnext-base_32xb128_in1k.py",
            "weightsUrl": "https://download.openmmlab.com/mmclassification/v0/convnext/convnext-base_32xb128_in1k_20221207-fbdb5eb9.pth",
            "model": "ConvNeXt Base",
            "collection": "ConvNeXt",
            "params": "88.59",
            "flops": "15.36",
            "top1": "83.66",
            "top5": "96.74",
        },
        {
            "modelConfig": "configs/_base_/models/convnext/convnext-large.py",
            "config": "configs/convnext/convnext-large_64xb64_in1k.py",
            "weightsUrl": "https://download.openmmlab.com/mmclassification/v0/convnext/convnext-large_3rdparty_64xb64_in1k_20220124-f8a0ded0.pth",
            "model": "ConvNeXt Large",
            "collection": "ConvNeXt",
            "params": "197.77",
            "flops": "34.37",
            "top1": "84.30",
            "top5": "96.89",
        },
        {
            "modelConfig": "configs/_base_/models/convnext/convnext-xlarge.py",
            "config": "configs/convnext/convnext-xlarge_64xb64_in1k.py",
            "weightsUrl": "https://download.openmmlab.com/mmclassification/v0/convnext/convnext-xlarge_in21k-pre-3rdparty_64xb64_in1k_20220124-76b6863d.pth",
            "model": "ConvNeXt XLarge",
            "collection": "ConvNeXt",
            "params": "350.20",
            "flops": "60.93",
            "top1": "86.97",
            "top5": "98.20",
        },
        # ConvNeXt V2
        {
            "modelConfig": "configs/_base_/models/convnext_v2/nano.py",
            "config": "configs/convnext_v2/convnext-v2-nano_32xb32_in1k.py",
            "weightsUrl": "https://download.openmmlab.com/mmclassification/v0/convnext-v2/convnext-v2-nano_3rdparty-fcmae_in1k_20230104-3dd1f29e.pth",
            "model": "ConvNeXtV2 Nano",
            "collection": "ConvNeXt",
            "params": "15.62",
            "flops": "2.45",
            "top1": "81.86",
            "top5": "95.75",
        },
        {
            "modelConfig": "configs/_base_/models/convnext_v2/tiny.py",
            "config": "configs/convnext_v2/convnext-v2-tiny_32xb32_in1k.py",
            "weightsUrl": "https://download.openmmlab.com/mmclassification/v0/convnext-v2/convnext-v2-tiny_3rdparty-fcmae_in1k_20230104-80513adc.pth",
            "model": "ConvNeXtV2 Tiny",
            "collection": "ConvNeXt",
            "params": "28.64",
            "flops": "4.47",
            "top1": "82.94",
            "top5": "96.29",
        },
        {
            "modelConfig": "configs/_base_/models/convnext_v2/base.py",
            "config": "configs/convnext_v2/convnext-v2-base_32xb32_in1k.py",
            "weightsUrl": "https://download.openmmlab.com/mmclassification/v0/convnext-v2/convnext-v2-base_3rdparty-fcmae_in1k_20230104-8a798eaf.pth",
            "model": "ConvNeXtV2 Base",
            "collection": "ConvNeXt",
            "params": "88.72",
            "flops": "15.38",
            "top1": "84.87",
            "top5": "97.08",
        },
        {
            "modelConfig": "configs/_base_/models/convnext_v2/large.py",
            "config": "configs/convnext_v2/convnext-v2-large_32xb32_in1k.py",
            "weightsUrl": "https://download.openmmlab.com/mmclassification/v0/convnext-v2/convnext-v2-large_3rdparty-fcmae_in1k_20230104-bf38df92.pth",
            "model": "ConvNeXtV2 Large",
            "collection": "ConvNeXt",
            "params": "197.96",
            "flops": "34.40",
            "top1": "85.76",
            "top5": "97.59",
        },
        {
            "modelConfig": "configs/_base_/models/convnext_v2/huge.py",
            "config": "configs/convnext_v2/convnext-v2-huge_32xb32_in1k.py",
            "weightsUrl": "https://download.openmmlab.com/mmclassification/v0/convnext-v2/convnext-v2-huge_3rdparty-fcmae_in1k_20230104-fe43ae6c.pth",
            "model": "ConvNeXtV2 Huge",
            "collection": "ConvNeXt",
            "params": "660.29",
            "flops": "115.00",
            "top1": "86.25",
            "top5": "97.75",
        },
        # MViT V2
        {
            "modelConfig": "configs/_base_/models/mvit/mvitv2-tiny.py",
            "config": "configs/mvit/mvitv2-tiny_8xb256_in1k.py",
            "weightsUrl": "https://download.openmmlab.com/mmclassification/v0/mvit/mvitv2-tiny_3rdparty_in1k_20220722-db7beeef.pth",
            "model": "MViT Tiny",
            "collection": "MViT",
            "params": "24.17",
            "flops": "4.70",
            "top1": "82.33",
            "top5": "96.15",
        },
        {
            "modelConfig": "configs/_base_/models/mvit/mvitv2-small.py",
            "config": "configs/mvit/mvitv2-small_8xb256_in1k.py",
            "weightsUrl": "https://download.openmmlab.com/mmclassification/v0/mvit/mvitv2-small_3rdparty_in1k_20220722-986bd741.pth",
            "model": "MViT Small",
            "collection": "MViT",
            "params": "34.87",
            "flops": "7.00",
            "top1": "83.63",
            "top5": "96.51",
        },
        {
            "modelConfig": "configs/_base_/models/mvit/mvitv2-base.py",
            "config": "configs/mvit/mvitv2-base_8xb256_in1k.py",
            "weightsUrl": "https://download.openmmlab.com/mmclassification/v0/mvit/mvitv2-base_3rdparty_in1k_20220722-9c4f0a17.pth",
            "model": "MViT Base",
            "collection": "MViT",
            "params": "51.47",
            "flops": "10.16",
            "top1": "84.34",
            "top5": "96.86",
        },
        {
            "modelConfig": "configs/_base_/models/mvit/mvitv2-large.py",
            "config": "configs/mvit/mvitv2-large_8xb256_in1k.py",
            "weightsUrl": "https://download.openmmlab.com/mmclassification/v0/mvit/mvitv2-large_3rdparty_in1k_20220722-2b57b983.pth",
            "model": "MViT Large",
            "collection": "MViT",
            "params": "217.99",
            "flops": "43.87",
            "top1": "85.25",
            "top5": "97.14",
        },
    ]
    _validate_models_configs(res)
    return res


def get_table_columns():
    return [
        {"key": "model", "title": "Model", "subtitle": None},
        {"key": "params", "title": "Params (M)", "subtitle": None},
        {"key": "flops", "title": "Flops (G)", "subtitle": None},
        {"key": "top1", "title": "Top-1 (%)", "subtitle": None},
        {"key": "top5", "title": "Top-5 (%)", "subtitle": None},
    ]


def get_model_info_by_name(name):
    models = get_models_list()
    for info in models:
        if info["model"] == name:
            return info
    raise KeyError(f"Model {name} not found")


def get_pretrained_weights_by_name(name):
    return get_model_info_by_name(name)["weightsUrl"]


def _validate_models_configs(models):
    return models
    res = []
    for model in models:
        model_config_path = os.path.join(g.root_source_dir, model["modelConfig"])
        train_config_path = os.path.join(g.root_source_dir, model["config"])
        if not sly.fs.file_exists(model_config_path):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), model_config_path)
        if not sly.fs.file_exists(train_config_path):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), train_config_path)
        res.append(model)
    return res


def init(data, state):
    models = get_models_list()
    data["models"] = models
    data["modelColumns"] = get_table_columns()
    state["selectedModel"] = "ResNet-34"  # "ResNet-50"
    state["weightsInitialization"] = "imagenet"  # "custom"  # "imagenet" #@TODO: for debug
    state["collapsed6"] = True
    state["disabled6"] = True
    init_progress(6, data)

    state["weightsPath"] = (
        ""  # "/mmclassification/5687_synthetic products v2_003/checkpoints/epoch_10.pth"  #@TODO: for debug
    )
    data["done6"] = False


def restart(data, state):
    data["done6"] = False
    # state["collapsed6"] = True
    # state["disabled6"] = True


@g.my_app.callback("download_weights")
@sly.timeit
@g.my_app.ignore_errors_and_show_dialog_window()
def download_weights(api: sly.Api, task_id, context, state, app_logger):
    global local_weights_path
    try:
        if state["weightsInitialization"] == "custom":
            weights_path_remote = state["weightsPath"]
            if not weights_path_remote.endswith(".pth"):
                raise ValueError(
                    f"Weights file has unsupported extension {sly.fs.get_file_ext(weights_path_remote)}. "
                    f"Supported: '.pth'"
                )

            # get architecture type from previous UI state
            prev_state_path_remote = os.path.join(
                str(Path(weights_path_remote).parents[1]), "info/ui_state.json"
            )
            prev_state_path = os.path.join(g.my_app.data_dir, "ui_state.json")
            api.file.download(g.team_id, prev_state_path_remote, prev_state_path)
            prev_state = sly.json.load_json_file(prev_state_path)
            api.task.set_field(g.task_id, "state.selectedModel", prev_state["selectedModel"])

            local_weights_path = os.path.join(
                g.my_app.data_dir, sly.fs.get_file_name_with_ext(weights_path_remote)
            )
            if sly.fs.file_exists(local_weights_path) is False:
                file_info = g.api.file.get_info_by_path(g.team_id, weights_path_remote)
                if file_info is None:
                    raise FileNotFoundError(
                        errno.ENOENT, os.strerror(errno.ENOENT), weights_path_remote
                    )
                progress_cb = get_progress_cb(
                    6, "Download weights", file_info.sizeb, is_size=True, min_report_percent=1
                )
                g.api.file.download(
                    g.team_id, weights_path_remote, local_weights_path, g.my_app.cache, progress_cb
                )
                reset_progress(6)
        else:
            weights_url = get_pretrained_weights_by_name(state["selectedModel"])
            local_weights_path = os.path.join(
                g.my_app.data_dir, sly.fs.get_file_name_with_ext(weights_url)
            )
            if sly.fs.file_exists(local_weights_path) is False:
                response = requests.head(weights_url, allow_redirects=True)
                sizeb = int(response.headers.get("content-length", 0))
                progress_cb = get_progress_cb(
                    6, "Download weights", sizeb, is_size=True, min_report_percent=1
                )
                sly.fs.download(weights_url, local_weights_path, g.my_app.cache, progress_cb)
                reset_progress(6)
        sly.logger.info(
            "Pretrained weights has been successfully downloaded",
            extra={"weights": local_weights_path},
        )
    except Exception as e:
        reset_progress(6)
        raise e

    g.local_weights_path = local_weights_path
    fields = [
        {"field": "data.done6", "payload": True},
        {"field": "state.collapsed7", "payload": False},
        {"field": "state.disabled7", "payload": False},
        {"field": "state.activeStep", "payload": 7},
    ]
    if state["selectedModel"].startswith("VGG"):
        fields.append({"field": "state.disabledImgSize", "payload": True})
    g.api.app.set_fields(g.task_id, fields)
