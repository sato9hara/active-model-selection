#!/bin/bash

for s in logreg ensemble; do
    for t in rf mlp; do
        for d in covtype letter mnist sensorless; do
            for m in uniform sawade proposed; do
                python test.py $d --general --method $m --model $t --surrogate $s --loss zo --n_test 1000 --seed 0
            done
        done
    done
done
