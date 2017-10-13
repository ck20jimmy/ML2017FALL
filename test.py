import sys
import numpy as np
import scipy
import pandas as pd
import csv


infile = sys.argv[1]
outfile = sys.argv[2]


w = pd.read_csv("param.csv").values

testdata = pd.read_csv('infile',header=None).values.tolist()

X = []

for sample in range(240):
	x_data = []
	for factor in range(18):
		for item in testdata[sample*18+factor][2:]:
			if item == 'NR':
				item = 0.0
			x_data.append(float(item))
	X.append(x_data)

#add 2
X = np.array(X)

X = np.concatenate((X,X**2),axis=1)



#add bias
#X = np.concatenate((np.ones((X.shape[0],1)),X), axis=1)
X = np.matrix(X)

y = np.dot(X,w)


ans = open(outfile,'w')
ans.write("id,value\n")
for row in range(240):	
	ans.write("id_"+str(row)+","+str(y[row,0])+"\n")

ans.close()
