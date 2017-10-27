import numpy as np
import pandas as pd
import sys

xfile = sys.argv[1]
yfile = sys.argv[2]

X_data = pd.read_csv(xfile)
Y_data = pd.read_csv(yfile)

dropfeature = pd.read_csv("drop_feature_new.csv",header=None).values.flatten().tolist()

X = np.array(X_data.drop(dropfeature,axis=1))
Y = np.array(Y_data)

"""
#normalize
mean = np.mean(X,axis=0)
std = np.std(X,axis=0)

X = X.astype(float)
X -= mean
X /= std
"""

X_train = X
X_test = X

Y_train = Y
Y_test = Y


Epoch = 10000

lr = 10**(-5)*6

#w = np.zeros((X_train[0].size,1))

w = pd.read_csv("logistic_param_regu.csv",header=None).values

g = np.zeros((X_train[0].size,1))

ld = 0.1

for i in range(Epoch):

	z = np.dot(X_train,w)
	
	f = 1/(1+np.exp(-z))

	L = -np.sum(np.multiply(Y_train,np.log(f))+np.multiply((1-Y_train),np.log(1-f)))

	grad = (-(np.dot(X_train.transpose(),(Y_train-f))))+2*ld*w

	g += grad**2

	adagrad = np.sqrt(g)

	#w -= lr*grad

	w -= lr*np.divide(grad,adagrad)

	print("Epoch%d: L = %f" % (i,L),end=" ")


	#test
	z_test = np.dot(X_test,w)
	f_test = (1/(1+np.exp(-z_test))).flatten().tolist()

	y = []

	for item in f_test:
		if (item < 0.5):
			y.append(0)
		else:
			y.append(1)

	ac = 0.0
	for index in range(len(y)):
		if(y[index] == Y_test[index,]):
			ac += 1

	ac /= len(f_test)
	print("accuracy: %f" %(ac))

'''
param = open("logistic_param.csv","w")

for p in w.flatten().tolist():
	param.write(str(p)+"\n")
param.close()
'''
