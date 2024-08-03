from torchvision.models import alexnet, convnext_base, efficientnet_v2_m, inception_v3, resnet152, swin_b, vgg16, vit_b_16, wide_resnet101_2

nets = {
    'AlexNet': (alexnet, 'AlexNet_Weights.IMAGENET1K_V1'), 
    'ConvNeXt': (convnext_base, 'ConvNeXt_Base_Weights.IMAGENET1K_V1'),
    'EfficientNet': (efficientnet_v2_m, 'EfficientNet_V2_M_Weights.IMAGENET1K_V1'),
    'InceptionV3': (inception_v3, 'Inception_V3_Weights.IMAGENET1K_V1'), 
    'ResNet': (resnet152, 'ResNet152_Weights.IMAGENET1K_V1'), 
    'SwinTransformer': (swin_b, 'Swin_B_Weights.IMAGENET1K_V1'),
    'VGG': (vgg16, 'VGG16_Weights.IMAGENET1K_V1'),
    'VisionTransformer': (vit_b_16, 'ViT_B_16_Weights.IMAGENET1K_V1'),
    'WideResNet': (wide_resnet101_2, 'Wide_ResNet101_2_Weights.IMAGENET1K_V1')
}

for key, (fn, weights) in nets.items():
    net = fn(weights=weights)
