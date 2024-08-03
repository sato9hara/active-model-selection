#!/bin/bash

wget -nc https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/covtype.bz2
wget -nc https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/mnist.bz2
wget -nc https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/letter.scale
wget -nc https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/Sensorless
bzip2 -dk *.bz2
mv letter.scale letter
mv Sensorless sensorless