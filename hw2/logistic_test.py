import numpy as np
import pandas as pd
import sys

xfile = sys.argv[1]

X_test = pd.read_csv(xfile)

dropfeature = pd.read_csv("drop_feature_new.csv",header=None).values.flatten().tolist()

X_test = X_test.drop(dropfeature,axis=1)

param = pd.read_csv("logistic_param_regu.csv",header=None)

f = np.dot(X_test,param).flatten().tolist()


ansfile = sys.argv[2]
ans = open(ansfile,"w")
ans.write("id,label\n")

for index in range(len(f)):
	if f[index] > 0.5:
		ans.write(str(index+1)+","+str(1)+"\n")
	else:
		ans.write(str(index+1)+","+str(0)+"\n")
