import sys
import numpy as np
import scipy
import pandas as pd
import tensorflow


#from numpy.linalg import inv

data = pd.read_csv('train.csv',sep=',',encoding='big5').values.tolist()

trainData = [ [] for x in range(12) ]

for month in range(12):
	for f in range(18):
		trainData[month].append([])
		for day in range(20):
			trainData[month][f] += data[month*360+day*18+f][3:]

feature = [9]#range(18)#[2,5,7,8,9,10,12,17]



'''
0,6,9 = 6.4185
0,6,9,10 = 6.405

0,6,9,10,12 = 6.38

0,6,9,10,12,17 = 6.380340	

best: 9, 平方, 20000 iteration

0-18 = 5.87...

2,6,7,9,10,12,13,17 = 5.814319

6,7,9,10,12,13,17 = 5.825377

2,6,7,8,9,10,12,13,17 = 5.761153

2,5,7,8,9,10,12,13,17 = 5.7501,	50000 = 5.742805

2,3,5,7,8,9,10,12,17 = 5.75089

'''


Y = []
X = []

hours = 9

for month in range(12):
	for start in range(480-hours):
		sample_x = []
		for f in range(18):
			for hour in range(hours):
				tmp = trainData[month][f][start+hour]
				if tmp == 'NR':
					sample_x.append(0.0)
				else:
					sample_x.append(float(tmp))
		
		Y.append(float(trainData[month][9][start+hours]))
		X.append(sample_x)



X = np.array(X)



xtmp = X[:,feature[0]*hours:(feature[0]+1)*hours]

for i in range(1,len(feature)):
	xtmp = np.concatenate((xtmp,X[:,feature[i]*hours:(feature[i]+1)*hours]),axis=1)

#add X^2
xtmp = np.concatenate((xtmp,xtmp**2),axis=1)

#add bias
#xtmp = np.concatenate((np.ones((xtmp.shape[0],1)),xtmp), axis=1)

X = np.matrix(xtmp)



Y = np.matrix(Y).transpose()


#X = np.concatenate((np.ones((X.shape[0],1)),X), axis=1)

'''
#calculate coef between each feature and PM2.5

coef = []

for col in range(X[0,:].size):
	tmpx = X[:,col].transpose()
	tmpy = Y.transpose()
	
	coef.append(np.corrcoef(tmpx,tmpy)[1,0])
	
	#cor.append(np.correlate(X[:][col],Y).tolist())

for idx, value in enumerate(coef):
	if abs(value) < 10**(-2):
		print( "feature %d %d-th factor coef = %f" % ( idx/9, idx%9, value ) )


sys.exit()
'''


'''
w = np.matmul(np.matmul(inv(np.matmul(X.transpose(),X)),X.transpose()),Y)
print(w)
#np.savetxt("best_param.csv",w,delimiter=',',header='0')
sys.exit()
'''

#print(len(X))
#print(len(Y))

#sys.exit()

lr_init = 10.0#0.0101
lr = np.full((X[0,:].size,1),lr_init)
lr = np.matrix(lr)

#lr[17*9:,] = 10**(1.5)

#lr[0] = 1
#lr[1:9] = 24

#print(lr)

#sys.exit()

#blr = 10**(-2)

w0 = 0.0
w = np.full((X[0,:].size,1),w0)
w = np.matrix(w)


#b_init = 0.0
#b = np.full((len(X),1),b_init)
#b = np.matrix(b)

g = np.full((X[0,:].size,1),0.0)
g = np.matrix(g)

#gb = np.full((len(X),1),0.0)

XT = X.transpose()
iteration = 10000

#reg_rate = 0.001

for i in range(iteration):

	y = np.dot(X,w)#+np.random.rand(len(Y),1)

	loss = Y - y

	#reg = reg_rate*((np.square(w)).sum())

	L = np.square(loss).sum()/len(X)

	grad_w = (-2)*XT*loss

	#grad_b = (-2)*(loss.sum())
	
	g = g + np.square(grad_w)
	#gb = gb + np.square(grad_b)

	ada = np.sqrt(g)
	#ada_b = np.sqrt(gb)
	
	'''
	for row in range(len(ada)):
		if ada[row,0] == 0:
			ada[row,0] = 1.0
	'''

	w = w - np.divide(np.multiply(lr,grad_w),ada)
	#b = b - grad_b/ada_b


	print("iteration %d | Cost = %f" % (i+1,np.sqrt(L)))

print(w)
print(feature)


param = []

#bias
#param.append(w[0,0])

offset = 0
for z in range(2):
	for i in range(18):
		if i in feature:
			for j in range(9-hours):
				param.append(0.0)

			for j in range(hours):
				param.append(w[offset*hours+j,0])
			offset += 1
		else:
			for j in range(9):
				param.append(0.0)

param = np.array(param)


np.savetxt("Report\\param.csv",param,delimiter=',',header='0')