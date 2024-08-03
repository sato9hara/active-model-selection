import sys
sys.path.append('../src')

import pathlib
import pickle
import json
import joblib
import argparse

import numpy as np

from config import get_model_config, get_method_config
from loss import log_loss, zero_one_loss, topk_loss
from query import sample_from_p

# load training results
def load_train(dn, nets, loss_fn):
    Lte = []
    for net in nets:
        fn = dn.joinpath('%s.pkl' % (net,))
        with open(str(fn), 'rb') as f:
            yte, pte, _ = pickle.load(f)
        pte = pte.astype(float)
        Lte.append(loss_fn(pte))
    Lte = np.array(Lte)
    Ltrue = np.array([Lte[:, i, c] for i, c in enumerate(yte)]).T
    return Ltrue, Lte

# surrogate
def pi_by_ensemble(dn, nets):
    pi = []
    for net in nets:
        fn = dn.joinpath('%s.pkl' % (net,))
        with open(str(fn), 'rb') as f:
            _, pte, _ = pickle.load(f)
        pte = pte.astype(float)
        pi.append(pte)
    pi = np.mean(np.array(pi), axis=0)
    return pi

# test model selection
def run_model_selection(Ltrue, Lte, p_est, q_fn, est_fn, n_query=1000, replace=False, seed=0):
    Ltrue = Ltrue.copy()
    Lte = Lte.copy()
    p_est = p_est.copy()
    
    # data
    Nte = Lte.shape[1]
    idx_q = [] # indices of queried points
    idx_u = list(range(Nte)) # indices of unqueried test points

    # initialization
    L, q, hat = [], [], []
    
    # for loop
    q_full = q_fn(Lte, p_est, idx_u, normalize=False)
    for i in range(n_query):
        
        # q
        q_now = q_full[idx_u]
        q_now = q_now / np.sum(q_now)
        
        # sampling from q_now
        k = sample_from_p(q_now, seed=Nte*seed+i+1)
        j = idx_u[k] # j: queried test index
        
        # observe the loss of j
        L.append([LL[j] for LL in Ltrue])
        q.append(q_now[k])
        
        # estimated average test loss
        hat.append(est_fn(np.array(L), np.array(q), Nte))

        # update queried and unqueried points
        if replace is False:
            idx_u.pop(k)
            idx_q.append(j)
    return np.array(hat)

def test(dir_out, dir_pred, method, loss_fn, n_query=1000, n_test=100):

    # config
    nets = get_model_config()
    method_config = get_method_config()

    # load train
    Ltrue, Lte = load_train(dir_pred, nets, loss_fn)
    Lavg = np.mean(Ltrue, axis=1)
    Lmin = np.min(Lavg)
    opt = np.where(Lavg == Lmin)[0]
    others = np.setdiff1d(range(Lavg.size), opt)

    # surrogate estimates
    p_est = pi_by_ensemble(dir_pred, nets)
    
    # run active testing
    (q_fn, est_fn, replace) = method_config[method]
    test_fn = lambda seed: run_model_selection(Ltrue, Lte, p_est, q_fn, est_fn, n_query=n_query, replace=replace, seed=seed)
    res = joblib.Parallel(n_jobs=-1)(joblib.delayed(test_fn)(seed) for seed in range(n_test))
    hat = np.array(res)

    # eval 1: success rate
    hmin = np.min(hat, axis=2)
    success = (np.sum((hat == hmin[:, :, np.newaxis])[:, :, others], axis=2) == 0)
    success = np.mean(success, axis=0)

    # eval 2: # of queries for identification
    q_num = []
    for h in res:
        idx = np.argmin(h, axis=1)
        for i, v in enumerate(idx[::-1]):
            if v not in opt:
                break
        q_num.append(len(idx) - i)

    # save
    fn = dir_out.joinpath('%s.pkl' % (method,))
    with open(str(fn), 'wb') as f:
        pickle.dump({'Lavg':Lavg,
                     'opt':(opt, others),
                     'hat': hat,
                     'success': success, 
                     'q_num': q_num},
                     f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data', type=str, choices=['mat', 'thr', 'top'])
    parser.add_argument('--method', type=str, choices=['uniform', 'sawade', 'proposed'])
    parser.add_argument('--loss', type=str, choices=['zo', 'log', 'top5'])
    parser.add_argument('--n_query', type=int, default=1000)
    parser.add_argument('--n_test', type=int, default=1000)
    args = parser.parse_args()

    # loss
    if args.loss == 'zo':
        loss_fn = zero_one_loss
    elif args.loss == 'log':
        loss_fn = log_loss
    elif args.loss == 'top5':
        loss_fn = lambda p: topk_loss(p, k=5)
    
    # output directory
    with open('../config.json') as f:
        config = json.load(f)
    dir_out = pathlib.Path(config['dir']).joinpath('Sec6/res/%s/%s' % (args.data, args.loss))
    dir_out.mkdir(parents=True, exist_ok=True)

    # pred directory
    dir_pred = pathlib.Path(config['dir']).joinpath('Sec6/preds/%s' % (args.data,))

    # test
    test(dir_out, 
         dir_pred, 
         args.method, 
         loss_fn, 
         n_query=args.n_query,
         n_test=args.n_test)
    
