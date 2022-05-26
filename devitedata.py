import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import csv

def Dive_Data(urlz, nor, pr):
	"""
	:param urlz: location
	:param nor:normalization
	:param pr:present
	:param pre:noisy
	:return:
	"""
	df = pd.read_csv(urlz, header=None)
	data = df.values
	numberSample, numberAttribute = data.shape

	if nor==True:
		minMax = MinMaxScaler()  # normalization
		U = np.hstack((minMax.fit_transform(data[0:numberSample, 1:]), data[0:numberSample, 0].reshape(numberSample, 1)))
	else:
		U = np.hstack(((data[0:numberSample, 1:]), data[0:numberSample, 0].reshape(numberSample, 1)))
	for i in range(len(U)):
		if U[i][-1] != 1:
			U[i][-1] = -1
	np.random.shuffle(U)
	U = U.tolist()
	train=U[0:int(numberSample*(1-pr))]
	test=U[int(numberSample * (1-pr)):]
	return train,test
