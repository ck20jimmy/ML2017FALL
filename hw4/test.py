import pandas as pd
import numpy as np
import sys
import os

from gensim.models import Word2Vec

from keras.models import Sequential, load_model
from keras.layers import Dense, GRU, Embedding, Dropout, Bidirectional
from keras.preprocessing import text, sequence
from keras import optimizers
from keras import regularizers
from keras import callbacks

tfile = sys.argv[1]
outfile = sys.argv[2]

testdata = pd.read_csv(tfile,sep='\n').values.tolist()

test_x = []

for row in testdata:
	index = row[0].find(',')
	st = row[0][index+1:].lower()
	test_x.append(st.split())

emb_model = Word2Vec.load("emb_model.bin")

weights = emb_model.wv.syn0

word2idx = {"_PAD":0}

vlist = [(k,emb_model.wv[k]) for k, v in emb_model.wv.vocab.items()]

emb_matrix = np.zeros((len(emb_model.wv.vocab.items())+1,emb_model.vector_size))

for i in range(len(vlist)):
    word = vlist[i][0]
    word2idx[word] = i+1
    emb_matrix[i+1] = vlist[i][1]

test_xdata = []

for st in test_x:
	tmp = []
	for word in st:
		tmp2 = 9999999
		if word in word2idx:
			tmp2 = word2idx[word]
		tmp.append(tmp2)
	test_xdata.append(tmp)

maxlen = 60
test_xdata = sequence.pad_sequences(test_xdata,maxlen)


model = Sequential()

batch = 2048

model.add(Embedding(input_dim=emb_matrix.shape[0],output_dim=emb_matrix.shape[1],weights=[emb_matrix],trainable=False))

model.add(Bidirectional(GRU(units=128,recurrent_regularizer=regularizers.l2(0.01),return_sequences=True)))
model.add(Dropout(0.2))

model.add(Bidirectional(GRU(units=128,recurrent_regularizer=regularizers.l2(0.01),return_sequences=True)))
model.add(Dropout(0.2))

model.add(Bidirectional(GRU(units=128,recurrent_regularizer=regularizers.l2(0.01))))
model.add(Dropout(0.2))

model.add(Dense(units=1,activation='sigmoid',kernel_regularizer=regularizers.l2(0.01)))

model.load_weights("bestmodel_weights.h5",by_name=True)

ans = model.predict(test_xdata)


output = open(outfile,'w+')
output.write("id,label\n")

for i in range(ans.shape[0]):
	tmp = 0
	if ans[i] > 0.5:
		tmp = 1
	output.write(str(i)+","+str(tmp)+'\n')


