import pandas as pd
import numpy as np
import sys
import os
import shutil

from gensim.models import Word2Vec

from keras.models import Sequential, load_model
from keras.layers import Dense, CuDNNGRU, CuDNNLSTM, Embedding, LSTM, Dropout, Bidirectional, StackedRNNCells, LSTMCell, RNN, BatchNormalization
from keras.preprocessing import text, sequence
from keras import optimizers
from keras import regularizers
from keras import callbacks

tlfile = sys.argv[1]
tnlfile = sys.argv[2]

tldata = pd.read_csv(tlfile,header=None,sep='\n').values

xtldata = []
ytldata = []
for item in tldata[:,0].tolist():
    ytldata.append(int(item[0]))
    tmp = item[10:].lower()
    xtldata.append(tmp.split())

ytldata = np.array(ytldata)

'''
testdata = pd.read_csv('testing_data.txt',sep='\n').values.tolist()

test_x = []

for row in testdata:
	index = row[0].find(',')
	st = row[0][index+1:].lower()
	test_x.append(st.split())
'''

xtul = pd.read_csv(tnlfile,header=None,sep='\n').values.tolist()

xtuldata = []

for item in xtul:
	tmp = item[0].lower()
	xtuldata.append(tmp.split())

#wordvec = xtldata + test_x + xtuldata

#emb_model = Word2Vec(wordvec,size=512)

#emb_model.save('emb_no_p_model.bin')

emb_model = Word2Vec.load("emb_model.bin")

weights = emb_model.wv.syn0

word2idx = {"_PAD":0}

vlist = [(k,emb_model.wv[k]) for k, v in emb_model.wv.vocab.items()]

emb_matrix = np.zeros((len(emb_model.wv.vocab.items())+1,emb_model.vector_size))

for i in range(len(vlist)):
    word = vlist[i][0]
    word2idx[word] = i+1
    emb_matrix[i+1] = vlist[i][1]

xtrain = []
for st in xtldata:
	tmp = []
	for word in st:
		if word not in word2idx:
			tmp2 = 9999999
		else:
			tmp2 = word2idx[word]
		tmp.append(tmp2)
	xtrain.append(tmp)

maxlen = 60

xtrain = sequence.pad_sequences(xtrain,maxlen)

model = Sequential()

batch = 2048

model.add(Embedding(input_dim=emb_matrix.shape[0],output_dim=emb_matrix.shape[1],weights=[emb_matrix],trainable=False))

model.add(Bidirectional(CuDNNGRU(units=128,recurrent_regularizer=regularizers.l2(0.01),return_sequences=True,name="d1")))
model.add(Dropout(0.2))

model.add(Bidirectional(CuDNNGRU(units=128,recurrent_regularizer=regularizers.l2(0.01),return_sequences=True,name="d2")))
model.add(Dropout(0.2))

model.add(Bidirectional(CuDNNGRU(units=128,recurrent_regularizer=regularizers.l2(0.01),name="d3")))
model.add(Dropout(0.2))

model.add(Dense(units=1,activation='sigmoid',kernel_regularizer=regularizers.l2(0.01),name="d4"))

adam = optimizers.Adam(lr=0.002,clipvalue=3)

nadam = optimizers.Nadam(lr=0.001,clipvalue=3)

rmsprop = optimizers.RMSprop(lr=0.0003,clipvalue=3)

model.compile(loss='binary_crossentropy',optimizer=adam,metrics=['accuracy'])

modeldir = "best_model"

if os.path.exists(modeldir):
	shutil.rmtree(modeldir)

os.mkdir(modeldir)


mcp = callbacks.ModelCheckpoint(modeldir+"/weights_{epoch:02d}-{val_loss:.2f}.hdf5", monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)

estp = callbacks.EarlyStopping(monitor='val_loss',patience=5)

rlop = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,patience=1, min_lr=10**(-10))

tb = callbacks.TensorBoard(log_dir="./logs")

cb = [rlop,estp,mcp]

model.fit(xtrain,ytldata,batch_size=batch, epochs=50,validation_split=0.1,callbacks=cb)


'''
model = load_model("model/kaggle_8270.hdf5")

ulx = []

for st in xtuldata:
    tmp = []
    for word in st:
        tmp2 = 9999999
        if word in word2idx:
            tmp2 = word2idx[word]
        tmp.append(tmp2)
    ulx.append(tmp)


ulx = sequence.pad_sequences(ulx,maxlen)

ans = model.predict(ulx)

np.save("ul_ydata",ans)
'''
sys.exit()



'''
test_xdata = []

for st in test_x:
	tmp = []
	for word in st:
		tmp2 = 9999999
		if word in word2idx:
			tmp2 = word2idx[word]
		tmp.append(tmp2)
	test_xdata.append(tmp)

test_xdata = sequence.pad_sequences(test_xdata,maxlen)

ans = model.predict(test_xdata)

output = open('no_punc_ans.csv','w+')
output.write("id,label\n")

for i in range(ans.shape[0]):
	tmp = 0
	if ans[i] > 0.5:
		tmp = 1
	output.write(str(i)+","+str(tmp)+'\n')
'''


'''

512

4 bidirec cudnnlstm



'''
