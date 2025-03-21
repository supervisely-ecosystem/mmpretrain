<div align="center" markdown>

<img src="https://user-images.githubusercontent.com/106374579/186635809-18aaf088-44a0-4ccc-8e68-f0ac97ce603b.png"/>

# Train MMClassification V2 (MMPretrain)

<p align="center">
  <a href="#Overview">Overview</a> •
  <a href="#Updates">Updates</a> •
  <a href="#How-To-Run">How To Run</a> •
  <a href="#How-To-Use">How To Use</a>
  <a href="#How-To-Use-Your-Trained-Model-Outside-Supervisely">How To Use Your Trained Model Outside Supervisely</a> •
</p>

[![](https://img.shields.io/badge/supervisely-ecosystem-brightgreen)](https://ecosystem.supervisely.com/apps/supervisely-ecosystem/mmpretrain/supervisely/train)
[![](https://img.shields.io/badge/slack-chat-green.svg?logo=slack)](https://supervisely.com/slack)
![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/supervisely-ecosystem/mmpretrain)
[![views](https://app.supervisely.com/img/badges/views/supervisely-ecosystem/mmpretrain/supervisely/train.png)](https://supervisely.com)
[![runs](https://app.supervisely.com/img/badges/runs/supervisely-ecosystem/mmpretrain/supervisely/train.png)](https://supervisely.com)

</div>

# Overview

Train models from [MMPretrain](https://github.com/open-mmlab/mmpretrain) (updated [MMClassification](https://github.com/open-mmlab/mmclassification)) toolbox on your custom data (Supervisely format is supported).
- Configure Train / Validation splits, model architecture and training hyperparameters
- Visualize and validate training data
- App automatically generates training py configs in MMPretrain format
- Run on any computer with GPU (agent) connected to your team
- Monitor progress, metrics, logs and other visualizations withing a single dashboard

## Model Zoo

| Model                   | Params (M) | Flops (G) | Top1  | Top5  | Скачать                                                                                                                                                                                                                                                                               |
|-------------------------|------------|-----------|-------|-------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| VGG-11                  | 132.86     | 7.63      | 68.75 | 88.87 | [config](https://github.com/open-mmlab/mmpretrain/blob/main/configs/vgg/vgg11_8xb32_in1k.py) \| [модель](https://download.openmmlab.com/mmclassification/v0/vgg/vgg11_batch256_imagenet_20210208-4271cd6c.pth)                                                                        |
| VGG-13                  | 133.05     | 11.34     | 70.02 | 89.46 | [config](https://github.com/open-mmlab/mmpretrain/blob/main/configs/vgg/vgg13_8xb32_in1k.py) \| [модель](https://download.openmmlab.com/mmclassification/v0/vgg/vgg13_batch256_imagenet_20210208-4d1d6080.pth)                                                                        |
| VGG-16                  | 138.36     | 15.5      | 71.62 | 90.49 | [config](https://github.com/open-mmlab/mmpretrain/blob/main/configs/vgg/vgg16_8xb32_in1k.py) \| [модель](https://download.openmmlab.com/mmclassification/v0/vgg/vgg16_batch256_imagenet_20210208-db26f1a5.pth)                                                                        |
| VGG-19                  | 143.67     | 19.67     | 72.41 | 90.80 | [config](https://github.com/open-mmlab/mmpretrain/blob/main/configs/vgg/vgg19_8xb32_in1k.py) \| [модель](https://download.openmmlab.com/mmclassification/v0/vgg/vgg19_bn_batch256_imagenet_20210208-da620c4f.pth)                                                                     |
| VGG-11-BN               | 132.87     | 7.64      | 70.75 | 90.12 | [config](https://github.com/open-mmlab/mmpretrain/blob/main/configs/vgg/vgg11bn_8xb32_in1k.py) \| [модель](https://download.openmmlab.com/mmclassification/v0/vgg/vgg11_bn_batch256_imagenet_20210207-f244902c.pth)                                                                   |
| VGG-13-BN               | 133.05     | 11.36     | 72.15 | 90.71 | [config](https://github.com/open-mmlab/mmpretrain/blob/main/configs/vgg/vgg13bn_8xb32_in1k.py) \| [модель](https://download.openmmlab.com/mmclassification/v0/vgg/vgg13_bn_batch256_imagenet_20210207-1a8b7864.pth)                                                                   |
| VGG-16-BN               | 138.37     | 15.53     | 73.72 | 91.68 | [config](https://github.com/open-mmlab/mmpretrain/blob/main/configs/vgg/vgg16bn_8xb32_in1k.py) \| [модель](https://download.openmmlab.com/mmclassification/v0/vgg/vgg16_bn_batch256_imagenet_20210208-7e55cd29.pth)                                                                   |
| VGG-19-BN               | 143.68     | 19.7      | 74.70 | 92.24 | [config](https://github.com/open-mmlab/mmpretrain/blob/main/configs/vgg/vgg19bn_8xb32_in1k.py) \| [модель](https://download.openmmlab.com/mmclassification/v0/vgg/vgg19_bn_batch256_imagenet_20210208-da620c4f.pth)                                                                   |
| ResNet-18               | 11.69      | 1.82      | 70.07 | 89.44 | [config](https://github.com/open-mmlab/mmpretrain/blob/main/configs/resnet/resnet18_8xb32_in1k.py) \| [модель](https://download.openmmlab.com/mmclassification/v0/resnet/resnet18_batch256_imagenet_20200708-34ab8f90.pth)                                                            |
| ResNet-34               | 21.8       | 3.68      | 73.85 | 91.53 | [config](https://github.com/open-mmlab/mmpretrain/blob/main/configs/resnet/resnet34_8xb32_in1k.py) \| [модель](https://download.openmmlab.com/mmclassification/v0/resnet/resnet34_batch256_imagenet_20200708-32ffb4f7.pth)                                                            |
| ResNet-50               | 25.56      | 4.12      | 76.55 | 93.15 | [config](https://github.com/open-mmlab/mmpretrain/blob/main/configs/resnet/resnet50_8xb32_in1k.py) \| [модель](https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_batch256_imagenet_20200708-cfb998bf.pth)                                                            |
| ResNet-101              | 44.55      | 7.85      | 78.18 | 94.03 | [config](https://github.com/open-mmlab/mmpretrain/blob/main/configs/resnet/resnet101_8xb32_in1k.py) \| [модель](https://download.openmmlab.com/mmclassification/v0/resnet/resnet101_batch256_imagenet_20200708-753f3608.pth)                                                          |
| ResNet-152              | 60.19      | 11.58     | 78.63 | 94.16 | [config](https://github.com/open-mmlab/mmpretrain/blob/main/configs/resnet/resnet152_8xb32_in1k.py) \| [модель](https://download.openmmlab.com/mmclassification/v0/resnet/resnet152_batch256_imagenet_20200708-ec25b1f9.pth)                                                          |
| ResNetV1D-50            | 25.58      | 4.36      | 77.54 | 93.57 | [config](https://github.com/open-mmlab/mmpretrain/blob/main/configs/resnet/resnetv1d50_8xb32_in1k.py) \| [модель](https://download.openmmlab.com/mmclassification/v0/resnet/resnetv1d50_b32x8_imagenet_20210531-db14775a.pth)                                                         |
| ResNetV1D-101           | 44.57      | 8.09      | 78.93 | 94.48 | [config](https://github.com/open-mmlab/mmpretrain/blob/main/configs/resnet/resnetv1d101_8xb32_in1k.py) \| [модель](https://download.openmmlab.com/mmclassification/v0/resnet/resnetv1d101_b32x8_imagenet_20210531-6e13bcd3.pth)                                                       |
| ResNetV1D-152           | 60.21      | 11.82     | 79.41 | 94.7  | [config](https://github.com/open-mmlab/mmpretrain/blob/main/configs/resnet/resnetv1d152_8xb32_in1k.py) \| [модель](https://download.openmmlab.com/mmclassification/v0/resnet/resnetv1d152_b32x8_imagenet_20210531-278cf22a.pth)                                                       |
| ResNeXt-32x4d-50        | 25.03      | 4.27      | 77.90 | 93.66 | [config](https://github.com/open-mmlab/mmpretrain/blob/main/configs/resnext/resnext50-32x4d_8xb32_in1k.py) \| [модель](https://download.openmmlab.com/mmclassification/v0/resnext/resnext50_32x4d_b32x8_imagenet_20210429-56066e27.pth)                                               |
| ResNeXt-32x4d-101       | 44.18      | 8.03      | 78.71 | 94.12 | [config](https://github.com/open-mmlab/mmpretrain/blob/main/configs/resnext/resnext101-32x4d_8xb32_in1k.py) \| [модель](https://download.openmmlab.com/mmclassification/v0/resnext/resnext101_32x4d_b32x8_imagenet_20210506-e0fa3dd5.pth)                                             |
| ResNeXt-32x8d-101       | 88.79      | 16.5      | 79.23 | 94.58 | [config](https://github.com/open-mmlab/mmpretrain/blob/main/configs/resnext/resnext101-32x8d_8xb32_in1k.py) \| [модель](https://download.openmmlab.com/mmclassification/v0/resnext/resnext101_32x8d_b32x8_imagenet_20210506-23a247d5.pth)                                             |
| ResNeXt-32x4d-152       | 59.95      | 11.8      | 78.93 | 94.41 | [config](https://github.com/open-mmlab/mmpretrain/blob/main/configs/resnext/resnext152-32x4d_8xb32_in1k.py) \| [модель](https://download.openmmlab.com/mmclassification/v0/resnext/resnext152_32x4d_b32x8_imagenet_20210524-927787be.pth)                                             |
| SE-ResNet-50            | 28.09      | 4.13      | 77.74 | 93.84 | [config](https://github.com/open-mmlab/mmpretrain/blob/main/configs/seresnet/seresnet50_8xb32_in1k.py) \| [модель](https://download.openmmlab.com/mmclassification/v0/se-resnet/se-resnet50_batch256_imagenet_20200804-ae206104.pth)                                                  |
| SE-ResNet-101           | 49.33      | 7.86      | 78.26 | 94.07 | [config](https://github.com/open-mmlab/mmpretrain/blob/main/configs/seresnet/seresnet101_8xb32_in1k.py) \| [модель](https://download.openmmlab.com/mmclassification/v0/se-resnet/se-resnet101_batch256_imagenet_20200804-ba5b51d4.pth)                                                |
| ShuffleNetV1 1.0x       | 1.87       | 0.146     | 68.13 | 87.81 | [config](https://github.com/open-mmlab/mmpretrain/blob/main/configs/shufflenet_v1/shufflenet-v1-1x_16xb64_in1k.py) \| [модель](https://download.openmmlab.com/mmclassification/v0/shufflenet_v1/shufflenet_v1_batch1024_imagenet_20200804-5d6cec73.pth)                               |
| ShuffleNetV2 1.0x       | 2.28       | 0.149     | 69.55 | 88.92 | [config](https://github.com/open-mmlab/mmpretrain/blob/main/configs/shufflenet_v2/shufflenet-v2-1x_16xb64_in1k.py) \| [модель](https://download.openmmlab.com/mmclassification/v0/shufflenet_v2/shufflenet_v2_batch1024_imagenet_20200812-5bf4721e.pth)                               |
| MobileNet V2            | 3.5        | 0.319     | 71.86 | 90.42 | [config](https://github.com/open-mmlab/mmpretrain/blob/main/configs/mobilenet_v2/mobilenet-v2_8xb32_in1k.py) \| [модель](https://download.openmmlab.com/mmclassification/v0/mobilenet_v2/mobilenet_v2_batch256_imagenet_20200708-3b2dc3af.pth)                                        |
| MobileNet V3 Small      | 2.54       | 0.06      | 66.68 | 86.74 | [config](https://github.com/open-mmlab/mmpretrain/blob/main/configs/mobilenet_v3/mobilenet-v3-small_8xb128_in1k.py) \| [модель](https://download.openmmlab.com/mmclassification/v0/mobilenet_v3/mobilenet-v3-small_8xb128_in1k_20221114-bd1bfcde.pth)                                 |
| MobileNet V3 Large      | 5.48       | 0.23      | 73.49 | 91.31 | [config](https://github.com/open-mmlab/mmpretrain/blob/main/configs/mobilenet_v3/mobilenet-v3-large_8xb128_in1k.py) \| [модель](https://download.openmmlab.com/mmclassification/v0/mobilenet_v3/mobilenet-v3-large_8xb128_in1k_20221114-0ed9ed9a.pth)                                 |
| ViT-B/16*               | 86.86      | 33.03     | 84.20 | 97.18 | [config](https://github.com/open-mmlab/mmpretrain/blob/main/configs/vision_transformer/vit-base-p16_64xb64_in1k-384px.py) \| [модель](https://download.openmmlab.com/mmclassification/v0/vit/finetune/vit-base-p16_in21k-pre-3rdparty_ft-64xb64_in1k-384_20210928-98e8652b.pth)       |
| ViT-B/32*               | 88.3       | 8.56      | 81.73 | 96.13 | [config](https://github.com/open-mmlab/mmpretrain/blob/main/configs/vision_transformer/vit-base-p32_64xb64_in1k-384px.py) \| [модель](https://download.openmmlab.com/mmclassification/v0/vit/finetune/vit-base-p32_in21k-pre-3rdparty_ft-64xb64_in1k-384_20210928-9cea8599.pth)       |
| ViT-L/16*               | 304.72     | 116.68    | 85.08 | 97.38 | [config](https://github.com/open-mmlab/mmpretrain/blob/main/configs/vision_transformer/vit-large-p16_64xb64_in1k-384px.py) \| [модель](https://download.openmmlab.com/mmclassification/v0/vit/finetune/vit-large-p16_in21k-pre-3rdparty_ft-64xb64_in1k-384_20210928-b20ba619.pth)     |
| EfficientNetV2 Small    | 21.46      | 9.72      | 84.29 | 97.26 | [config](https://github.com/open-mmlab/mmpretrain/blob/main/configs/efficientnet_v2/efficientnetv2-s_8xb32_in21k.py) \| [модель](https://download.openmmlab.com/mmclassification/v0/efficientnetv2/efficientnetv2-s_3rdparty_in21k_20221220-c0572b56.pth)                             |
| EfficientNetV2 Medium   | 54.14      | 26.88     | 85.47 | 97.76 | [config](https://github.com/open-mmlab/mmpretrain/blob/main/configs/efficientnet_v2/efficientnetv2-m_8xb32_in21k.py) \| [модель](https://download.openmmlab.com/mmclassification/v0/efficientnetv2/efficientnetv2-m_3rdparty_in21k_20221220-073e944c.pth)                             |
| EfficientNetV2 Large    | 118.52     | 60.14     | 86.31 | 97.99 | [config](https://github.com/open-mmlab/mmpretrain/blob/main/configs/efficientnet_v2/efficientnetv2-l_8xb32_in21k.py) \| [модель](https://download.openmmlab.com/mmclassification/v0/efficientnetv2/efficientnetv2-l_3rdparty_in21k_20221220-f28f91e1.pth)                             |
| EfficientNetV2 XLarge   | 208.12     | 98.34     | 86.39 | 97.83 | [config](https://github.com/open-mmlab/mmpretrain/blob/main/configs/efficientnet_v2/efficientnetv2-xl_8xb32_in21k.py) \| [модель](https://download.openmmlab.com/mmclassification/v0/efficientnetv2/efficientnetv2-xl_3rdparty_in21k_20221220-b2c9329c.pth)                           |
| SwinTransformerV2 Base  | 87.92      | 15.14     | 86.17 | 97.88 | [config](https://github.com/open-mmlab/mmpretrain/blob/main/configs/swin_transformer_v2/swinv2-base-w16_in21k-pre_16xb64_in1k-256px.py) \| [модель](https://download.openmmlab.com/mmclassification/v0/swin-v2/swinv2-base-w16_in21k-pre_3rdparty_in1k-256px_20220803-8d7aa8ad.pth)   |
| SwinTransformerV2 Large | 196.75     | 33.86     | 86.93 | 98.06 | [config](https://github.com/open-mmlab/mmpretrain/blob/main/configs/swin_transformer_v2/swinv2-large-w16_in21k-pre_16xb64_in1k-256px.py) \| [модель](https://download.openmmlab.com/mmclassification/v0/swin-v2/swinv2-large-w16_in21k-pre_3rdparty_in1k-256px_20220803-c40cbed7.pth) |
| DeiT3 Small             | 22.06      | 4.61      | 81.35 | 95.31 | [config](https://github.com/open-mmlab/mmpretrain/blob/main/configs/deit3/deit3-small-p16_64xb64_in1k.py) \| [модель](https://download.openmmlab.com/mmclassification/v0/deit3/deit3-small-p16_3rdparty_in1k_20221008-0f7c70cf.pth)                                                   |
| DeiT3 Medium            | 38.85      | 8.00      | 82.99 | 96.22 | [config](https://github.com/open-mmlab/mmpretrain/blob/main/configs/deit3/deit3-medium-p16_64xb64_in1k.py) \| [модель](https://download.openmmlab.com/mmclassification/v0/deit3/deit3-medium-p16_3rdparty_in1k_20221008-3b21284d.pth)                                                 |
| DeiT3 Base              | 86.59      | 17.58     | 83.80 | 96.55 | [config](https://github.com/open-mmlab/mmpretrain/blob/main/configs/deit3/deit3-base-p16_64xb64_in1k.py) \| [модель](https://download.openmmlab.com/mmclassification/v0/deit3/deit3-base-p16_3rdparty_in1k_20221008-60b8c8bf.pth)                                                     |
| DeiT3 Large             | 304.37     | 61.60     | 84.87 | 97.01 | [config](https://github.com/open-mmlab/mmpretrain/blob/main/configs/deit3/deit3-large-p16_64xb64_in1k.py) \| [модель](https://download.openmmlab.com/mmclassification/v0/deit3/deit3-large-p16_3rdparty_in1k_20221009-03b427ea.pth)                                                   |
| DeiT3 Huge              | 632.13     | 167.40    | 85.21 | 97.36 | [config](https://github.com/open-mmlab/mmpretrain/blob/main/configs/deit3/deit3-huge-p14_64xb32_in1k.py) \| [модель](https://download.openmmlab.com/mmclassification/v0/deit3/deit3-huge-p14_3rdparty_in1k_20221009-e107bcb7.pth)                                                     |
| ConvNeXt Tiny           | 28.59      | 4.46      | 82.14 | 96.06 | [config](https://github.com/open-mmlab/mmpretrain/blob/main/configs/convnext/convnext-tiny_32xb128_in1k.py) \| [модель](https://download.openmmlab.com/mmclassification/v0/convnext/convnext-tiny_32xb128_in1k_20221207-998cf3e9.pth)                                                 |
| ConvNeXt Small          | 50.22      | 8.69      | 83.16 | 96.56 | [config](https://github.com/open-mmlab/mmpretrain/blob/main/configs/convnext/convnext-small_32xb128_in1k.py) \| [модель](https://download.openmmlab.com/mmclassification/v0/convnext/convnext-small_32xb128_in1k_20221207-4ab7052c.pth)                                               |
| ConvNeXt Base           | 88.59      | 15.36     | 83.66 | 96.74 | [config](https://github.com/open-mmlab/mmpretrain/blob/main/configs/convnext/convnext-base_32xb128_in1k.py) \| [модель](https://download.openmmlab.com/mmclassification/v0/convnext/convnext-base_32xb128_in1k_20221207-fbdb5eb9.pth)                                                 |
| ConvNeXt Large          | 197.77     | 34.37     | 84.30 | 96.89 | [config](https://github.com/open-mmlab/mmpretrain/blob/main/configs/convnext/convnext-large_64xb64_in1k.py) \| [модель](https://download.openmmlab.com/mmclassification/v0/convnext/convnext-large_3rdparty_64xb64_in1k_20220124-f8a0ded0.pth)                                        |
| ConvNeXt XLarge         | 350.20     | 60.93     | 86.97 | 98.20 | [config](https://github.com/open-mmlab/mmpretrain/blob/main/configs/convnext/convnext-xlarge_64xb64_in1k.py) \| [модель](https://download.openmmlab.com/mmclassification/v0/convnext/convnext-xlarge_in21k-pre-3rdparty_64xb64_in1k_20220124-76b6863d.pth)                            |
| ConvNeXtV2 Nano         | 15.62      | 2.45      | 81.86 | 95.75 | [config](https://github.com/open-mmlab/mmpretrain/blob/main/configs/convnext_v2/convnext-v2-nano_32xb32_in1k.py) \| [модель](https://download.openmmlab.com/mmclassification/v0/convnext-v2/convnext-v2-nano_3rdparty-fcmae_in1k_20230104-3dd1f29e.pth)                               |
| ConvNeXtV2 Tiny         | 28.64      | 4.47      | 82.94 | 96.29 | [config](https://github.com/open-mmlab/mmpretrain/blob/main/configs/convnext_v2/convnext-v2-tiny_32xb32_in1k.py) \| [модель](https://download.openmmlab.com/mmclassification/v0/convnext-v2/convnext-v2-tiny_3rdparty-fcmae_in1k_20230104-80513adc.pth)                               |
| ConvNeXtV2 Base         | 88.72      | 15.38     | 84.87 | 97.08 | [config](https://github.com/open-mmlab/mmpretrain/blob/main/configs/convnext_v2/convnext-v2-base_32xb32_in1k.py) \| [модель](https://download.openmmlab.com/mmclassification/v0/convnext-v2/convnext-v2-base_3rdparty-fcmae_in1k_20230104-8a798eaf.pth)                               |
| ConvNeXtV2 Large        | 197.96     | 34.40     | 85.76 | 97.59 | [config](https://github.com/open-mmlab/mmpretrain/blob/main/configs/convnext_v2/convnext-v2-large_32xb32_in1k.py) \| [модель](https://download.openmmlab.com/mmclassification/v0/convnext-v2/convnext-v2-large_3rdparty-fcmae_in1k_20230104-bf38df92.pth)                             |
| ConvNeXtV2 Huge         | 660.29     | 115.00    | 86.25 | 97.75 | [config](https://github.com/open-mmlab/mmpretrain/blob/main/configs/convnext_v2/convnext-v2-huge_32xb32_in1k.py) \| [модель](https://download.openmmlab.com/mmclassification/v0/convnext-v2/convnext-v2-huge_3rdparty-fcmae_in1k_20230104-fe43ae6c.pth)                               |
| MViT Tiny               | 24.17      | 4.70      | 82.33 | 96.15 | [config](https://github.com/open-mmlab/mmpretrain/blob/main/configs/mvit/mvitv2-tiny_8xb256_in1k.py) \| [модель](https://download.openmmlab.com/mmclassification/v0/mvit/mvitv2-tiny_3rdparty_in1k_20220722-db7beeef.pth)                                                             |
| MViT Small              | 34.87      | 7.00      | 83.63 | 96.51 | [config](https://github.com/open-mmlab/mmpretrain/blob/main/configs/mvit/mvitv2-small_8xb256_in1k.py) \| [модель](https://download.openmmlab.com/mmclassification/v0/mvit/mvitv2-small_3rdparty_in1k_20220722-986bd741.pth)                                                           |
| MViT Base               | 51.47      | 10.16     | 84.34 | 96.86 | [config](https://github.com/open-mmlab/mmpretrain/blob/main/configs/mvit/mvitv2-base_8xb256_in1k.py) \| [модель](https://download.openmmlab.com/mmclassification/v0/mvit/mvitv2-base_3rdparty_in1k_20220722-9c4f0a17.pth)                                                             |
| MViT Large              | 217.99     | 43.87     | 85.25 | 97.14 | [config](https://github.com/open-mmlab/mmpretrain/blob/main/configs/mvit/mvitv2-large_8xb256_in1k.py) \| [модель](https://download.openmmlab.com/mmclassification/v0/mvit/mvitv2-large_3rdparty_in1k_20220722-2b57b983.pth)                                                           |


Watch [how-to video](https://youtu.be/R9sbH3biCmQ) about similar [Train MMClassification](https://ecosystem.supervisely.com/apps/mmclassification/supervisely/train) app for details:

<a data-key="sly-embeded-video-link" href="https://youtu.be/R9sbH3biCmQ" data-video-code="R9sbH3biCmQ">
    <img src="https://i.imgur.com/O47n1S1.png" alt="SLY_EMBEDED_VIDEO_LINK"  style="max-width:100%;">
</a>

# How to Run

1. Be sure that you connected computer with GPU to your team by running Supervisely Agent on it ([how-to video](https://youtu.be/aDqQiYycqyk))
2. Run app from context menu of project with tagged images
<img src="https://i.imgur.com/qz7IsXF.png"/>
1. Open Training Dashboard (app UI) and follow instructions provided in the video above

# How To Use
1. App downloads input project from Supervisely Instance to the local directory
2. Define train / validation splits
   - Randomly
   <img src="https://i.imgur.com/mwcos1I.png"/>

   - Based on image tags (for example "train" and "val", you can assign them yourself)
   <img src="https://i.imgur.com/X9mnuRK.png"/>

   - Based on datasets (if you want to use some datasets for training (for example "ds0", "ds1", "ds3") and 
     other datasets for validation (for example "ds_val"), it is completely up to you)
     <img src="https://i.imgur.com/Y956BvC.png"/>

3. Preview all available tags with corresponding image examples. Select training tags (model will be trained to predict them).
   <img src="https://i.imgur.com/g7eY0AC.png"/>

4. App validates data consistency and correctness and produces report.
   <img src="https://i.imgur.com/AHExs93.png"/>

5. Select how to augment data. All augmentations performed on the fly during training.
   - use one of the predefined pipelines
   <img src="https://i.imgur.com/tJpY1uc.png"/>
   - or use custom augs.  To create custom augmentation pipeline use app
   "[ImgAug Studio](https://ecosystem.supervisely.com/apps/imgaug-studio)" from Supervisely Ecosystem. This app allows to 
   export pipeline in several formats. To use custom augs just provide the path to JSON config in team files.
     <a data-key="sly-embeded-video-link" href="https://youtu.be/ZkZ7krcKq1c" data-video-code="ZkZ7krcKq1c">
         <img src="https://i.imgur.com/HFEhrdl.png" alt="SLY_EMBEDED_VIDEO_LINK"  style="max-width:100%;">
     </a>
   - preview selected augs on the random image from project
     <img src="https://i.imgur.com/TwCJnmv.png"/>

6. Select model and how weights should be initialized
   - pretrained on imagenet
   <img src="https://i.imgur.com/LppcO7C.png"/>
   - custom weights, provide path to the weights file in team files
   <a data-key="sly-embeded-video-link" href="https://youtu.be/XU9vCwHh9_g" data-video-code="XU9vCwHh9_g">
     <img src="https://i.imgur.com/1YHXLty.png" alt="SLY_EMBEDED_VIDEO_LINK"  style="max-width:100%;">
   </a>

7. Configure training hyperparameters
   <img src="https://i.imgur.com/IW5ywEo.png"/>

8. App generates py configs for MMPretrain toolbox automatically. Just press `Generate` button and move forward.
   You can modify configuration manually if you are advanced user and understand MMToolBox.
   <img src="https://i.imgur.com/a87AR7A.png"/>

9. Start training
   <img src="https://i.imgur.com/NsPUbyF.png"/>
   
10. All training artifacts (metrics, visualizations, weights, ...) are uploaded to Team Files. Link to the directory
    is generated in UI after training.
   
   Save path is the following: ```"/mmclassification-v2/<task id>_<input project name>/```

   For example: ```/mmclassification-v2/5886_synthetic products v2/```
   
   Structure is the following:
   ```
   . 
   ├── checkpoints
   │   ├── best_accuracy_top-1_epoch_44.pth
   │   ├── epoch_48.pth
   │   ├── epoch_49.pth
   │   ├── epoch_50.pth
   │   └── latest.pth
   ├── configs
   │   ├── dataset_config.py
   │   ├── model_config.py
   │   ├── runtime_config.py
   │   ├── schedule_config.py
   │   └── train_config.py
   ├── info
   │   ├── gt_labels.json
   │   ├── tag2urls.json
   │   └── ui_state.json
   └── open_app.lnk
   ```
- `checkpoints` directory contains MMPretrain weights
- `configs` directory contains all configs that app generated for MMPretrain toolbox, they may be useful
for advanced user who would like ot export models and use them outside Supervisely
- info directory contains basic information about training
   - `gt_labels.json` - mapping between class names and their indices, this file allows to understand NN predictions. For examples:
   ```json
  {
      "cat": 0,
      "dog": 1,
      "bird": 2,
      "frog": 3
  }
   ```
  - `tag2urls.json` - for every tag some image examples were saved, this file is used when the model is integrated into labeling interface
   ```json
  {
     "cat": [
          "http://supervisely.private/a/b/c/01.jpg",
          "http://supervisely.private/a/b/c/02.jpg"
     ],
     "dog": [
         "http://supervisely.private/d/d/d/01.jpg",
         "http://supervisely.private/d/d/d/02.jpg"
     ],
     "bird": [
         "http://supervisely.private/c/c/c/01.jpg",
         "http://supervisely.private/c/c/c/02.jpg"
     ],
     "frog": [
         "http://supervisely.private/c/c/c/01.jpg",
         "http://supervisely.private/c/c/c/02.jpg"
     ]
  }
   ```
  - `ui_state.json` file with all values defined in UI
   ```json
   {
      "...": "...",
      "epochs": 50,
      "gpusId": "0",
      "imgSize": 256,
      "batchSizePerGPU": 64,
      "workersPerGPU": 3,
      "valInterval": 1,
      "metricsPeriod": 10,
      "checkpointInterval": 1,
      "maxKeepCkptsEnabled": true,
      "maxKeepCkpts": 3,
      "saveLast": true,
      "saveBest": true,
      "optimizer": "SGD",
      "lr": 0.001,
      "momentum": 0.9,
      "weightDecay": 0.0001,
      "....": "..."
   }
  ```   

- `open_app.lnk` - link to finished session, you can open finished training dashboard at any time and check all settings and visualizations
   <img src="https://i.imgur.com/BVtNo7E.png"/>

# How To Use Your Trained Model Outside Supervisely

You can use your trained models outside Supervisely platform without any dependencies on Supervisely SDK. You just need to download config files and model weights (.pth) from Team Files, and then you can build and use the model as a normal model in mmcls/mmpretrain.
