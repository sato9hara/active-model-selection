import pathlib
import json
import pickle
import argparse
from utils import load_data
from config import get_model_config

def train(dn, data, model, train_size=5000, test_size=500, random_state=0):

    # load data
    x, y, xte, yte = load_data(data, train_size=train_size, test_size=test_size, random_state=random_state)
    
    # fit models
    config = get_model_config()
    fit_fn, option = config[model]
    dnn = dn.joinpath(model)
    dnn.mkdir(exist_ok=True)
    for o in option:
        clf = fit_fn(x, y, option=o, random_state=random_state)
        z = clf.predict(x)
        p = clf.predict_proba(x)
        zte = clf.predict(xte)
        pte = clf.predict_proba(xte)
        fn = dnn.joinpath('p_opt%05d_seed%02d.pkl' % (o, random_state))
        with open(str(fn), 'wb') as f:
            pickle.dump((y, yte, z, zte, p, pte), f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data', type=str, choices=['covtype', 'letter', 'mnist', 'sensorless'])
    parser.add_argument('--model', type=str, choices=['rf', 'mlp', 'logreg'])
    parser.add_argument('--train_size', type=int, default=5000)
    parser.add_argument('--test_size', type=int, default=500)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    # output directory
    with open('../config.json') as f:
        config = json.load(f)
    dn = pathlib.Path(config['dir']).joinpath('preds/%s' % (args.data,))
    dn.mkdir(parents=True, exist_ok=True)

    # test
    train(dn,
          args.data, 
          args.model,
          train_size=args.train_size, 
          test_size=args.test_size, 
          random_state=args.seed)
