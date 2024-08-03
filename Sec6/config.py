import sys
sys.path.append('../src')
from query import q_by_loss_diff, q_by_loss_diff_squared, q_uniform
from estimator import LURE, WAVG, AVG

def get_model_config():
    config = {'AlexNet', 'ConvNeXt', 'EfficientNet', 'InceptionV3', 'ResNet', 
            'SwinTransformer', 'VGG', 'VisionTransformer', 'WideResNet'}
    return config

def get_method_config():
    config = {'uniform': (q_uniform, AVG, False), 
              'sawade': (q_by_loss_diff_squared, WAVG, True), 
              'proposed': (q_by_loss_diff, LURE, False)}
    return config
