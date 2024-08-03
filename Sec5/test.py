import sys
sys.path.append('../src')

import pathlib
import pickle
import json
import joblib
import argparse

import numpy as np

from config import get_method_config, get_test_config
from loss import log_loss, zero_one_loss
from query import sample_from_p

# load training results
def load_train(dn, model, option, loss_fn, random_state=0):
    dnn = dn.joinpath(model)
    Lte, zte = [], []
    for o in option:
        fn = dnn.joinpath('p_opt%05d_seed%02d.pkl' % (o, random_state))
        with open(str(fn), 'rb') as f:
            _, yte, _, _, _, pte = pickle.load(f)
        Lte.append(loss_fn(pte))
    Lte = np.array(Lte)
    Ltrue = np.array([Lte[:, i, c] for i, c in enumerate(yte)]).T
    return Ltrue, Lte

# surrogate
def pi_by_ensemble(dn, model, options, random_state=0):
    dnn = dn.joinpath(model)
    pi = []
    for o in options:
        fn = dnn.joinpath('p_opt%05d_seed%02d.pkl' % (o, random_state))
        with open(str(fn), 'rb') as f:
            _, _, _, _, _, pte = pickle.load(f)
        pi.append(pte)
    pi = np.mean(np.array(pi), axis=0)
    return pi

# test model selection
def run_model_selection(Ltrue, Lte, p_est, q_fn, est_fn, replace=False, seed=0):
    
    # data
    Nte = Lte.shape[1]
    idx_q = [] # indices of queried points
    idx_u = list(range(Nte)) # indices of unqueried test points

    # initialization
    L, q, hat = [], [], []
    
    # for loop
    q_full = q_fn(Lte, p_est, idx_u, normalize=False)
    for i in range(Nte):
        
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

def test(dir_out, dir_pred, method, model, surrogate, loss_fn, general=False, n_test=1000, random_state=0):

    # configs
    method_config = get_method_config()
    test_config = get_test_config(general=general)

    # load train
    Ltrue, Lte = load_train(dir_pred, model, test_config[model], loss_fn, random_state=random_state)
    Lavg = np.mean(Ltrue, axis=1)
    Lmin = np.min(Lavg)
    opt = np.where(Lavg == Lmin)[0]
    others = np.setdiff1d(range(Lavg.size), opt)

    # surrogate estimates
    if surrogate == 'logreg':
        p_est = pi_by_ensemble(dir_pred, 'logreg', [0], random_state=random_state)
    elif surrogate == 'ensemble':
        p_est = pi_by_ensemble(dir_pred, model, test_config[model], random_state=random_state)

    # run active testing
    (q_fn, est_fn, replace) = method_config[method]
    test_fn = lambda seed: run_model_selection(Ltrue, Lte, p_est, q_fn, est_fn, replace=replace, seed=seed)
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
    dnn = dir_out.joinpath(method)
    dnn.mkdir(exist_ok=True)
    fn = dnn.joinpath('seed%03d.pkl' % (random_state,))
    with open(str(fn), 'wb') as f:
        pickle.dump({'Lavg':Lavg,
                     'opt':(opt, others),
                     'hat': hat,
                     'success': success, 
                     'q_num': q_num},
                     f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data', type=str, choices=['covtype', 'letter', 'mnist', 'sensorless'])
    parser.add_argument('--method', type=str, choices=['uniform', 'diff', 'sawade', 'proposed'])
    parser.add_argument('--model', type=str, choices=['rf', 'mlp'])
    parser.add_argument('--surrogate', type=str, choices=['logreg', 'ensemble'])
    parser.add_argument('--loss', type=str, choices=['zo', 'log'])
    parser.add_argument('--general', action='store_true')
    parser.add_argument('--n_test', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    # loss
    if args.loss == 'zo':
        loss_fn = zero_one_loss
    elif args.loss == 'log':
        loss_fn = log_loss

    # output directory
    with open('../config.json') as f:
        config = json.load(f)
    dn = pathlib.Path(config['dir'])
    if args.general:
        dir_out = dn.joinpath('Sec53/%s/%s/%s/%s' % (args.data, args.model, args.surrogate, args.loss))
    else:
        dir_out = dn.joinpath('Sec52/%s/%s/%s/%s' % (args.data, args.model, args.surrogate, args.loss))
    dir_out.mkdir(parents=True, exist_ok=True)

    # pred directory
    dir_pred = dn.joinpath('preds/%s' % (args.data,))

    # test
    test(dir_out,
         dir_pred, 
         args.method, 
         args.model,
         args.surrogate,
         loss_fn, 
         general=args.general,
         n_test=args.n_test, 
         random_state=args.seed)
