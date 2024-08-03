#!/bin/bash

for n in AlexNet ConvNeXt EfficientNet InceptionV3 ResNet SwinTransformer VGG VisionTransformer WideResNet ; do
    python eval_models.py --data $1 --net $n --gpu $2
done
