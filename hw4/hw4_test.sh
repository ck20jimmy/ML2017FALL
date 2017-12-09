#!/bin/bash
wget https://www.dropbox.com/s/yo02gne5foihxal/bestmodel_weights.h5?dl=1
mv bestmodel_weights.h5?dl=1 ./bestmodel_weights.h5

wget https://www.dropbox.com/s/l9wppmtc28rs8yl/emb_model.bin?dl=1
mv ./emb_model.bin?dl=1 ./emb_model.bin

wget https://www.dropbox.com/s/yp040buywdoblqo/emb_model.bin.syn1neg.npy?dl=1
mv ./emb_model.bin.syn1neg.npy?dl=1 ./emb_model.bin.syn1neg.npy

wget https://www.dropbox.com/s/gr66g8d2635qhbz/emb_model.bin.wv.syn0.npy?dl=1
mv ./emb_model.bin.wv.syn0.npy?dl=1 ./emb_model.bin.wv.syn0.npy

python3 test.py $1 $2