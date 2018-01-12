#!/bin/bash
wget 
mv bestmodel_weights.h5?dl=1 ./bestmodel_weights.h5
wget
mv ./emb_model.bin?dl=1 ./emb_model.bin
wget
mv ./emb_model.bin.syn1neg.npy?dl=1 ./emb_model.bin.syn1neg.npy
wget
mv ./emb_model.bin.wv.syn0.npy?dl=1 ./emb_model.bin.wv.syn0.npy

python3 test.py $1 $2