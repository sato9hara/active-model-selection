#!/bin/bash

for m in rf mlp logreg; do
    for d in covtype letter mnist sensorless; do
        python train.py $d --model $m --train_size 5000 --test_size 500 --seed 0
    done
done
