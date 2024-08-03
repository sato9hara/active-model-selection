#!/bin/bash

for l in zo top5 log; do
    for d in mat thr top; do
        for m in uniform sawade proposed; do
            python test.py $d --method $m --loss $l --n_query 1000 --n_test 1000
        done
    done
done