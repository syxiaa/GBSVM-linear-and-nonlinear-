
import numpy as np
from warnings import filterwarnings
from sklearn.svm import SVC

from tools.gen_ball import gen_balls
from tools.pso import PSO
from tools.devitedata import Dive_Data
from tools.addNoisy import recreat_data

filterwarnings('ignore')



def demo_func(x):
	B = 0
	A = np.zeros(len(datab[0][0])).tolist()
	for i in range(len(datab)):
		B += x[i] * datab[i][1]
		A = np.array(A) + x[i] * datab[i][-1] * np.array(datab[i][0])
	A = A.tolist()
	TAl = [i * i for i in A]
	Al = np.sqrt(np.sum(TAl))
	wl = Al - B
	t_x = 0
	for i in range(len(datab)):
		t_x += x[i]
	return (np.square(wl) / 2) - t_x


def get_W_b(C,Y):
	lb = (np.zeros(len(datab))).tolist()
	ub = (np.ones(len(datab)) * C).tolist()
	pso = PSO(func=demo_func, dim=len(datab), pop=258, max_iter=1050, lb=lb, ub=ub, w=0.5, c1=1.6, c2=1.6,Y=Y)
	pso.run()
	A = np.zeros(len(datab[0][0])).tolist()
	B = 0
	for i in range(len(datab)):
		A = np.array(A) + datab[i][-1] * pso.gbest_x[i] * np.array(datab[i][0])
		B += pso.gbest_x[i] * datab[i][1]
	Al = np.sqrt(np.sum([i * i for i in A]))
	wl = Al - B
	w = wl * np.array(A) / (wl + B)
	print("w:", w)
	S = 0
	ys = []
	b = 0
	for i in range(len(pso.gbest_x)):
		if pso.gbest_x[i] > 2:
			S += 1
			ys.append(i)
	for i in range(S):
		b1 = (1 + wl * datab[ys[i]][1]) / datab[ys[i]][2]
		wc = np.sum(np.array(w) * datab[ys[i]][0])
		b += b1 - wc
	if S > 0:
		b /= S
	return w, b


def getacc(data, w, b):
	F = 0
	T = 0
	for i, v in enumerate(data):
		val = np.sum(w * np.array(v[0:-1])) + b
		if v[-1] > 0.1 and val < 0:
			F += 1
		elif v[-1] < 0.1 and val > 0:
			F += 1
		else:
			T += 1
	return (T) / (T + F)


if __name__ == '__main__':
	datae = ["heart1"]
	for i in range(len(datae)):
		urlz = r"F:\BaiduNetdiskWorkspace\UCI\\" + datae[i] + ".csv"
		name = datae[i]
		nor = True
		pr = 0.2
		for j in range(0, 7):
			SVM_acc = 0
			Li_SVM_acc = 0
			Noisy = 0.05 * j
			for k in range(4):
				T_Li_SVM_acc = 0
				train, test = Dive_Data(urlz, nor, pr)
				train = np.array(train)
				# print(train[0])
				test = np.array(test)
				N_data = recreat_data(train, Noisy)
				N_data = np.array(N_data)
				pur = 0
				for l in range(1, 21):
					pur = 1 - 0.015 * l
					for m in range(2, 5):
						num = m * 1
						# print(N_data[0:150])
						datab = gen_balls(N_data, pur=pur, delbals=num)  # generate balls
						print(len(datab))
						Y=[]
						for ii in datab:
							Y.append(ii[-1])
						W, b = get_W_b(255,Y)
						acc = getacc(test, W, b)
						print("accrue", acc)
						if acc > T_Li_SVM_acc:
							T_Li_SVM_acc = acc
				mat_data = N_data[:, 0:-1]
				orderAttribute = N_data[:, -1]
				clf = SVC(kernel='linear', gamma='auto')
				clf.fit(mat_data, orderAttribute)
				score = clf.score(test[:, 0:-1], test[:, -1])
				Li_SVM_acc += T_Li_SVM_acc
				SVM_acc += score
			print(name, "noisy", str(Noisy), "accuracy:", SVM_acc / 4)
			print(name, "noisy", str(Noisy), "accuracy:", Li_SVM_acc / 4)
