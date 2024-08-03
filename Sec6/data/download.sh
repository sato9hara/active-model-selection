#!/bin/bash

wget -nc https://huggingface.co/datasets/vaishaal/ImageNetV2/resolve/main/imagenetv2-matched-frequency.tar.gz &&
wget -nc https://huggingface.co/datasets/vaishaal/ImageNetV2/resolve/main/imagenetv2-threshold0.7.tar.gz &&
wget -nc https://huggingface.co/datasets/vaishaal/ImageNetV2/resolve/main/imagenetv2-top-images.tar.gz &&
tar -xvf imagenetv2-matched-frequency.tar.gz &&
tar -xvf imagenetv2-threshold0.7.tar.gz &&
tar -xvf imagenetv2-top-images.tar.gz