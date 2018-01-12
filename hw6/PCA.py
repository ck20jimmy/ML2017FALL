import numpy as np
from skimage import io
import sys
import os

filedir = sys.argv[1]
filename = sys.argv[2]

X = []
X_mean = np.zeros((1080000))

img_num = 0
for file in os.listdir(filedir):
	if file.endswith(".jpg"):
		img = io.imread(filedir+"/"+file)
		img = img.reshape(1080000)
		X.append(img)
		X_mean += img
		img_num += 1

X_mean /= img_num

X = np.stack(X)

X_PCA = (X - X_mean).transpose()

U,s,V = np.linalg.svd(X_PCA,full_matrices=False)

y = io.imread(filedir+"/"+filename).reshape(1080000)-X_mean

w = []
rec = np.zeros(1080000)
for i in range(4):
	tmp = np.dot(y,U[:,i])
	w.append(tmp)
	rec += tmp*U[:,i]

rec += X_mean
rec -= np.min(rec)
rec /= np.max(rec)
rec = (rec*255).astype(np.uint8)
rec = rec.reshape(600,600,3)

outfile = "reconstruction6.jpg"
io.imsave(outfile,rec)


