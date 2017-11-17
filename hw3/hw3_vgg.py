import numpy as np
import pandas as pd
import sys
import os
import random

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


#read training data
xfile = '../train.csv'
data = pd.read_csv(xfile)
xdata = data['feature'].values.tolist()
x = []
for index in range(len(xdata)):
	x.append(list(np.fromstring(xdata[index],dtype=int,sep=' ').reshape(48,48)))

xdata = np.stack(x,axis=0)
ydata = np.array(data.loc[:,'label'])

index = list(range(28709))
random.shuffle(index)
vxdata = xdata[index[:2871]]
vxdata = np.expand_dims(vxdata,axis=3)
vydata = np.take(ydata,index[:2871])
vydata = to_categorical(vydata,num_classes=7)

xdata = np.delete(xdata,index[:2871],0)
xdata = np.expand_dims(xdata,axis=3)

ydata = np.delete(ydata,index[:2871],0)
ydata = to_categorical(ydata,num_classes=7)



'''
datagen = ImageDataGenerator(horizonal_flip=True)
datagen.fit(xdata)
'''

#global val_loss
#val_loss = 0

#build model
model = Sequential()

# model.add(ZeroPadding2D((1,1),input_shape=(48,48,1)))
model.add(Conv2D(filters=64,kernel_size=(3,3),padding='same',input_shape=(48,48,1)))
model.add(normalization.BatchNormalization())
model.add(LeakyReLU(alpha=0.3))
# model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(filters=64,kernel_size=(3,3),padding='same'))
model.add(normalization.BatchNormalization())
model.add(LeakyReLU(alpha=0.3))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(normalization.BatchNormalization())
# model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(filters=128,kernel_size=(3,3),padding='same'))
model.add(normalization.BatchNormalization())
model.add(LeakyReLU(alpha=0.3))
# model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(filters=128,kernel_size=(3,3),padding='same'))
model.add(normalization.BatchNormalization())
model.add(LeakyReLU(alpha=0.3))
# model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(filters=128,kernel_size=(3,3),padding='same'))
model.add(normalization.BatchNormalization())
model.add(LeakyReLU(alpha=0.3))
model.add(MaxPooling2D(pool_size=(2,2)))


model.add(normalization.BatchNormalization())
# model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(filters=256,kernel_size=(3,3),padding='same'))
model.add(normalization.BatchNormalization())
model.add(LeakyReLU(alpha=0.3))
# model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(filters=256,kernel_size=(3,3),padding='same'))
model.add(normalization.BatchNormalization())
model.add(LeakyReLU(alpha=0.3))
# model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(filters=256,kernel_size=(3,3),padding='same'))
model.add(normalization.BatchNormalization())
model.add(LeakyReLU(alpha=0.3))
model.add(MaxPooling2D(pool_size=(2,2)))


model.add(normalization.BatchNormalization())
# model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(filters=512 ,kernel_size=(3,3),padding='same'))
model.add(normalization.BatchNormalization())
model.add(LeakyReLU(alpha=0.3))
model.add(Conv2D(filters=512 ,kernel_size=(3,3),padding='same'))
model.add(normalization.BatchNormalization())
model.add(LeakyReLU(alpha=0.3))
# model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(filters=512 ,kernel_size=(3,3),padding='same'))
model.add(normalization.BatchNormalization())
model.add(LeakyReLU(alpha=0.3))
model.add(MaxPooling2D(pool_size=(2,2)))


model.add(normalization.BatchNormalization())
# model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(filters=512 ,kernel_size=(3,3),padding='same'))
model.add(normalization.BatchNormalization())
model.add(LeakyReLU(alpha=0.3))
# model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(filters=512 ,kernel_size=(3,3),padding='same'))
model.add(normalization.BatchNormalization())
model.add(LeakyReLU(alpha=0.3))
# model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(filters=512 ,kernel_size=(3,3),padding='same'))
model.add(normalization.BatchNormalization())
model.add(LeakyReLU(alpha=0.3))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(units=4096))
model.add(normalization.BatchNormalization())
model.add(LeakyReLU(alpha=0.3))
model.add(Dropout(0.5))

model.add(Dense(units=4096))
model.add(normalization.BatchNormalization())
model.add(LeakyReLU(alpha=0.3))

model.add(Dense(units=2048))
model.add(normalization.BatchNormalization())
model.add(LeakyReLU(alpha=0.3))
# model.add(Dropout(0.5))

model.add(Dense(units=1024))
model.add(normalization.BatchNormalization())
model.add(LeakyReLU(alpha=0.3))
model.add(Dropout(0.3))

model.add(Dense(activation='softmax',units=7,kernel_regularizer=regularizers.l2(0.01)))

nadam = optimizers.Nadam(lr = 0.0005,beta_1=0.9,beta_2=0.999,epsilon=1e-08, schedule_decay=0.004)

adam = optimizers.Adam(lr=0.001,beta_1=0.9,beta_2=0.999,epsilon=1e-08,decay=0.0)

model.compile(loss='categorical_crossentropy',optimizer=nadam,metrics=['accuracy'])

batch_print_callback = LambdaCallback(
    on_epoch_end=lambda batch, logs: print(
        '\nINFO:root:Epoch[%d] Train-accuracy=%f\nINFO:root:Epoch[%d] Validation-accuracy=%f' %
        (batch, logs['acc'], batch, logs['val_acc'])))

''''
class estop(Callback):
	def __init__(self, monitor='val_loss', patience=5,verbose=0):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.value = value
        self.verbose = verbose
    def on_epoch_end(self, epoch, logs={}):
        if self.monitor < val_loss:
        	self.patience -= 1
        if self.patience == 0:
        	self.model.stop_training = True
'''

#estop = EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='auto')


tbcb = TensorBoard(log_dir='./Graph',histogram_freq=0, batch_size=32, write_graph=True, write_grads=False, write_images=True, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=32, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)

model.fit(x=xdata, y=ydata, validation_data=(vxdata,vydata),validation_split=0.0,epochs=100,batch_size=64,callbacks=[tbcb])

np.save('vxdata',vxdata)
np.save('vydata',vydata)


'''
def my_print(x):
	with open('model_summary.txt',"w") as fd:
		print(x,file=fd)

model.summary(print_fn=my_print)
'''

'''
model.fit_generator()
'''


#model.save('model/model_vgg.h5')


'''
valid ac = 0.54


'''
