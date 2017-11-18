import numpy as np
import pandas as pd
import sys
import os
from keras.models import load_model

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import normalization
from keras.layers import regularizers
from keras.utils import to_categorical
from keras import optimizers
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import LeakyReLU
from keras.layers import ZeroPadding2D

from keras.layers import Flatten
from keras.callbacks import LambdaCallback
from keras.callbacks import EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler
import keras.backend as K
from keras.callbacks import TensorBoard


xfile = sys.argv[1]
data = pd.read_csv(xfile)
xdata = data['feature'].values.tolist()


x = []
for index in range(len(xdata)):
	x.append(list(np.fromstring(xdata[index],dtype=int,sep=' ').reshape(48,48)))

xdata = np.stack(x,axis=0)
xdata = np.expand_dims(xdata,axis=3)

model = load_model("model_vgg.h5?dl=1")

res = model.predict(xdata)

outfile = sys.argv[2]
output = open(outfile,"w")

output.write("id,label\n")

for index in range(res.shape[0]):
	tmp = np.argmax(res[index])
	output.write(str(index)+","+str(tmp)+'\n')


