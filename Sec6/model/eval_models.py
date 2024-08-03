import argparse
import pathlib
import pickle
import json

import torch
from torchvision import transforms

from torchvision.models import alexnet, convnext_base, efficientnet_v2_m, inception_v3, resnet152, swin_b, vgg16, vit_b_16, wide_resnet101_2
from torchvision.datasets import ImageFolder

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

data = {
    'mat': '../data/imagenetv2-matched-frequency-format-val',
    'thr': '../data/imagenetv2-threshold0.7-format-val',
    'top': '../data/imagenetv2-top-images-format-val'
}

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

def get_loader(data_dir):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    test_transform = transforms.Compose([
        transforms.Resize((256, 256), antialias=True),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    dataset = ImageFolder(root=data_dir, transform=test_transform)
    i2c = {v:int(k) for k, v in dataset.class_to_idx.items()}
    dataset = ImageFolder(root=data_dir, transform=test_transform, target_transform=transforms.Lambda(lambda y: i2c[y]))
    loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=500, shuffle=False, num_workers=4, pin_memory=True)
    return loader

def eval_net(key, data_dir, out_dir, device=torch.device('cuda')):
    loader = get_loader(data_dir)
    net = nets[key][0](weights=nets[key][1]).eval().to(device)
    acc = 0
    Y, Z = [], []
    with torch.inference_mode():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            z = net(x)
            z = torch.softmax(z, dim=1)
            acc = acc + z.argmax(dim=1).eq(y).sum().item()
            Y.append(y.to('cpu'))
            Z.append(z.to('cpu'))
    Y, Z = torch.cat(Y).numpy(), torch.cat(Z).numpy().astype(float)
    fn = out_dir.joinpath(key+'.pkl')
    with open(str(fn), 'wb') as f:
        pickle.dump((Y, Z, acc), f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, choices=list(data.keys()))
    parser.add_argument('--net', type=str, choices=list(nets.keys()))
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    data_dir = data[args.data]
    with open('../../config.json') as f:
        config = json.load(f)
    out_dir = pathlib.Path(config['dir']).joinpath('Sec6/preds/%s' % (args.data,))
    out_dir.mkdir(parents=True, exist_ok=True)
    eval_net(args.net, data_dir, out_dir, device=torch.device('cuda:%d' % (args.gpu,)))
