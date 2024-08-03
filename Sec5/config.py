import sys
sys.path.append('../src')
from utils import fit_logreg, fit_random_forest, fit_mlp
from query import q_by_loss_diff, q_by_loss_diff_squared, q_uniform
from estimator import LURE, WAVG, AVG

def get_model_config():
    config = {'rf': (fit_random_forest, [10, 12, 14, 16, 18, 20]),
              'mlp': (fit_mlp, [30, 50, 100, 300, 500, 1000]), 
              'logreg': (fit_logreg, [0])}
    return config

def get_method_config():
    config = {'uniform': (q_uniform, AVG, False), 
              'sawade': (q_by_loss_diff_squared, WAVG, True), 
              'proposed': (q_by_loss_diff, LURE, False)}
    return config

def get_test_config(general=False):
    if not general:
        config = {'rf': [14, 20], 
                  'mlp': [100, 1000]}
    else:
        c = get_model_config()
        config = {'rf': c['rf'][1], 
                  'mlp': c['mlp'][1]}
    return config