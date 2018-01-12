import numpy as np
import pandas as pd
import sys

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, UpSampling2D, Reshape, BatchNormalization, Dropout
from keras.layers import LeakyReLU,Activation
from keras.models import Model, load_model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.optimizers import Nadam

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn import preprocessing


imgfile = sys.argv[1]
testfile = sys.argv[2]
outfile = sys.argv[3]


traindata = np.load(imgfile)
traindata = traindata.astype('float32')
# traindata = traindata / 255.0

traindata = preprocessing.scale(traindata)

traindata = traindata.reshape((traindata.shape[0],28,28,1))


'''
cnn1_nfilter,cnn2_nfilter,cnn3_nfilter = 20,40,20
cnn1_size,cnn2_size = 3,2

rdim = 64

#autoencoder model
input_img = Input(shape=(28,28,1,))
encoded = Conv2D(cnn1_nfilter,cnn1_size,padding='same')(input_img)
encoded = LeakyReLU()(encoded)
encoded = MaxPooling2D(pool_size=2,padding='same')(encoded)

encoded = Conv2D(cnn2_nfilter,cnn2_size,padding='same')(encoded)
encoded = LeakyReLU()(encoded)
encoded = MaxPooling2D(pool_size=2,padding='same')(encoded)

dse = Flatten()(encoded)

dse = Dense(units=1024)(dse)
dse = BatchNormalization()(dse)
dse = LeakyReLU()(dse)

dse = Dense(units=512)(dse)
dse = BatchNormalization()(dse)
dse = LeakyReLU()(dse)

#bottle neck
btn = Dense(units=rdim)(dse)

dse2 = Dense(units=cnn2_nfilter*7*7)(btn)
dse2 = BatchNormalization()(dse2)
dse2 = LeakyReLU()(dse2)

rs = Reshape((7,7,cnn2_nfilter))(dse2)

decoded = Conv2D(cnn2_nfilter,cnn2_size,padding='same')(rs)
decoded = LeakyReLU()(decoded)
# decoded = Activation('relu')(decoded)

decoded = UpSampling2D(size=(2,2))(decoded)

decoded = Conv2D(cnn1_nfilter,cnn1_size,padding='same')(decoded)
decoded = LeakyReLU()(decoded)
# decoded = Activation('relu')(decoded)

decoded = UpSampling2D(size=(2,2))(decoded)

decoded = Conv2D(1,2,padding='same')(decoded)
'''

'''

input_img = Input((784,))
dse = Dense(units=392,activation='relu')(input_img)

dse = Dense(units=196,activation='relu')(dse)

#bottle neck
btn = Dense(units=rdim)(dse)

ddse = Dense(units=196,activation='relu')(btn)

ddse = Dense(units=392,activation='relu')(ddse)

decoded = Dense(units=784)(ddse)

'''
'''
mdc = ModelCheckpoint("model/model.hdf5",monitor='val_loss',save_best_only=True,period=1)
estp = EarlyStopping(monitor='val_loss',patience=3)
rdlr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=1)

nadam = Nadam(lr=0.002)

autoencoder = Model(input_img,decoded)
encoder = Model(input_img,btn)

autoencoder.compile(optimizer=nadam,loss='mse')
autoencoder.fit(traindata,traindata,epochs=200,batch_size=128
	,validation_split=0.1,callbacks=[mdc,estp,rdlr])
'''
testdata = pd.read_csv(testfile).values

image1idx = testdata[:,1].tolist()
image2idx = testdata[:,2].tolist()

encoder = load_model("./best_encoder")

#get encoded data
rd = encoder.predict(traindata)
kmeans = KMeans(n_clusters = 2).fit(rd)
label = kmeans.labels_


with open(outfile,'w') as outfd:
	outfd.write("ID,Ans\n")
	
	for i in range(len(image1idx)):
		if label[image1idx[i]] == label[image2idx[i]]:
			outfd.write(str(i)+","+"1\n")
		else:
			outfd.write(str(i)+","+"0\n")
	
	outfd.close()


