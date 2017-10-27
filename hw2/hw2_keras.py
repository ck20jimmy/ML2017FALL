import numpy as np
import pandas as pd
import sys
from sklearn.metrics import matthews_corrcoef

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import normalization
from keras import regularizers


xfile = sys.argv[1]
yfile = sys.argv[2]

X_data = pd.read_csv(xfile)
Y_data = pd.read_csv(yfile)


Feature = list(X_data.columns)

dropfeature = pd.read_csv("drop_feature_new.csv",header=None).values.flatten().tolist()

X = np.array(X_data.drop(dropfeature,axis=1))
Y = np.array(Y_data)


model = Sequential()

model.add(Dense(60, input_shape=(X.shape[1],), activation='sigmoid'))

model.add(normalization.BatchNormalization())

model.add(Dense(40, activation='sigmoid',
	kernel_regularizer=regularizers.l2(0.01)))

model.add(Dense(1,activation='sigmoid'))

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.fit(X, Y,epochs=600, batch_size=9000, validation_split=0.33)

score = model.evaluate(X,Y)



testfile = sys.argv[3]

testdata = np.array(pd.read_csv(testfile).drop(dropfeature,axis=1))

ans = model.predict(testdata)




outfile = sys.argv[4]
output = open(outfile,'w')
output.write("id,label\n")


for row in range(len(ans)):	
	tmp = 0
	if ans[row,0] >= 0.5:
		tmp = 1

	output.write(str(row+1)+","+str(tmp)+"\n")

output.close()

