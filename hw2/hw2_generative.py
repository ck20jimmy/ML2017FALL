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

#normalize
mean = np.mean(X,axis=0)
std = np.std(X,axis=0)


X = X.astype(float)
X -= mean
X /= std


C1index = []
C2index = []

X_train = X#[:30000,:]
X_test = X#[30000:,:]


Y_train = Y#[:30000,:]
Y_test = Y#[30000:,:]

for index in range(Y_train.shape[0]):
	if Y_train[index] == 1:
		C1index.append(index)
	else:
		C2index.append(index)


C1 = np.array([ X_train[index,:] for index in C1index])
C2 = np.array([ X_train[index,:] for index in C2index])

N1 = len(C1)
N2 = len(C2)

u1 = np.mean(C1,axis=0)
u2 = np.mean(C2,axis=0)

sigma1 = C1 - u1
sigma1 = np.dot(sigma1.transpose(),sigma1)
sigma1 = np.divide(sigma1,N1)

sigma2 = C2 - u2
sigma2 = np.dot(sigma2.transpose(),sigma2)
sigma2 = np.divide(sigma2,N2)

sigma = sigma1*(N1/(N1+N2))+sigma2*(N2/(N1+N2))

sigma_inv = np.linalg.inv(sigma)


testfile = sys.argv[3]
X_test = pd.read_csv(testfile)
X_test = np.array(X_test.drop(dropfeature,axis=1))


#normalize
X_test = X_test.astype(float)
X_test -= mean
X_test /= std


z = np.dot((u1-u2),sigma_inv)
z = np.dot(z,X_test.transpose())

b = -(0.5)*np.dot(np.dot(u1,sigma_inv),u1.transpose()) + (0.5)*np.dot(np.dot(u2,sigma_inv),u2)+np.log(float(N1)/N2)

z = (z+b).transpose()

p = 1+np.exp(-z)

p = np.reciprocal(p).tolist()


ansfile = sys.argv[4]

ans = open(ansfile,"w")
ans.write("id,label\n")

for index in range(X_test.shape[0]):
	tmp = 0
	if p[index] > 0.5:
		tmp = 1
	ans.write(str(index+1)+","+str(tmp)+"\n")
