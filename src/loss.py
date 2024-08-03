import numpy as np

def log_loss(p, eps=1e-8):
    return - np.log(np.minimum(1-eps, np.maximum(eps, p)))

def zero_one_loss(p):
    return (p < np.max(p, axis=1)[:, np.newaxis]).astype(float)

def topk_loss(p, k=5):
    t = np.sort(p, axis=1)[:, -k]
    return (p < t[:, np.newaxis]).astype(float)
