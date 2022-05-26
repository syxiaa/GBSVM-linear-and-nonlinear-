import csv
import time
import pandas as pd
import numpy as np
import json
import csv
# import sklearn.cluster.k_means_ as KMeans
from sklearn.cluster import KMeans
import warnings
from collections import Counter
# from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

import pandas as pd

warnings.filterwarnings("ignore")  # ignore warning


class GranularBall:
    """class of the granular ball"""
    def __init__(self, data):
        """
        :param data:  Labeled data set, the "-2" column is the class label, the last column is the index of each line
        and each of the preceding columns corresponds to a feature
        """
        self.data = data[:, :]
        self.data_no_label = data[:, :-2]
        self.num, self.dim = self.data_no_label.shape
        self.center = self.data_no_label.mean(0)
        self.label, self.purity ,self.r= self.__get_label_and_purity_and_r()

    def __get_label_and_purity_and_r(self):
        """
        :return: the label and purity of the granular ball.
        """
        count = Counter(self.data[:, -2])
        label = max(count, key=count.get)
        purity = count[label] / self.num
        arr=np.array(self.data_no_label)-self.center
        ar=np.square(arr)
        a=np.sqrt(np.sum(ar,1))
        r=np.sum(a)/len(self.data_no_label)
        return label, purity,r

    def split_2balls(self):
        """
        split the granular ball to 2 new balls by using 2_means.
        """
        # label_cluster = KMeans(X=self.data_no_label, n_clusters=2)[1]
        clu=KMeans(n_clusters=2).fit(self.data_no_label)
        label_cluster=clu.labels_
        if sum(label_cluster == 0) and sum(label_cluster == 1):
            ball1 = GranularBall(self.data[label_cluster == 0, :])
            ball2 = GranularBall(self.data[label_cluster == 1, :])
        else:
            ball1 = GranularBall(self.data[0:1, :])
            ball2 = GranularBall(self.data[1:, :])
        return ball1, ball2


class GBList:
    """class of the list of granular ball"""
    def __init__(self, data=None):
        self.data = data[:, :]
        self.granular_balls = [GranularBall(self.data)]  # gbs is initialized with all data

    def init_granular_balls(self, purity=1, min_sample=1):
        """
        Split the balls, initialize the balls list.
        :param purity: If the purity of a ball is greater than this value, stop splitting.
        :param min_sample: If the number of samples of a ball is less than this value, stop splitting.
        """
        ll = len(self.granular_balls)
        i = 0
        while True:
            if self.granular_balls[i].purity < purity and self.granular_balls[i].num > min_sample:
                split_balls = self.granular_balls[i].split_2balls()
                self.granular_balls[i] = split_balls[0]
                self.granular_balls.append(split_balls[1])
                ll += 1
            else:
                i += 1
            if i >= ll:
                break
        self.data = self.get_data()

    def get_data_size(self):
        return list(map(lambda x: len(x.data), self.granular_balls))

    def get_purity(self):
        return list(map(lambda x: x.purity, self.granular_balls))

    def get_center(self):
        """
        :return: the center of each ball.
        """
        return np.array(list(map(lambda x: x.center, self.granular_balls)))

    def get_r(self):
        """
        :return: 返回半径r
        """
        return np.array(list(map(lambda x: x.r, self.granular_balls)))

    def get_data(self):
        """
        :return: Data from all existing granular balls in the GBlist.
        """
        list_data = [ball.data for ball in self.granular_balls]
        return np.vstack(list_data)
    def del_ball(self,purty=0.,num_data=0):
        T_ball=[]
        for ball in self.granular_balls:
            if ball.purity>=purty and ball.num>=num_data:
                T_ball.append(ball)
        self.granular_balls=T_ball.copy()
        self.data=self.get_data()
    def re_division(self, i):
        """
        Data division with the center of the ball.
        :return: a list of new granular balls after divisions.
        """
        k = len(self.granular_balls)
        attributes = list(range(self.data.shape[1] - 2))
        attributes.remove(i)
        clu = KMeans(n_clusters=k, init=self.get_center()[:, attributes], max_iter=1).fit(self.data[:, attributes])
        label_cluster = clu.labels_
        # label_cluster = KMeans(X=self.data[:, attributes], n_clusters=k,
        #                         init=self.get_center()[:, attributes], max_iter=1)[1]
        granular_balls_division = []
        for i in set(label_cluster):
            granular_balls_division.append(GranularBall(self.data[label_cluster == i, :]))
        return granular_balls_division
def generate_ball_data(data,pur,delbals):
    num, dim = data[:, :-1].shape
    index = np.array(range(num)).reshape(num, 1)  # column of index
    data = np.hstack((data, index))  # Add the index column to the last column of the data
    # step 1.
    print(data[0:4])
    gb = GBList(data)  # create the list of granular balls
    gb.init_granular_balls(purity=pur)  # initialize the list
    gb.del_ball(num_data=delbals)
    centers=gb.get_center().tolist()
    rs=gb.get_r().tolist()
    # print(type(centers[0]))
    balldata = []  # 检验
    for i in range(len(gb.granular_balls)):
        a=[]
        a.append(centers[i])
        a.append(rs[i])
        # print(data[i][-2])
        if gb.granular_balls[i].label==-1:
            a.append(-1)
        elif gb.granular_balls[i].label==1:
            a.append(1)
        balldata.append(a)
    # print(balldata[0])
    return balldata
def gen_balls(data,pur,delbals):
    # df=pd.read_csv(url,header=None)
    # data=df.values
    # print(data.shape)
    balls=generate_ball_data(data,pur=pur,delbals=delbals)
    R_balls=[]
    for i in balls:
        t_ball=[]
        t_ball.append(i[0])
        t_ball.append(i[1])
        t_ball.append(i[2])
        R_balls.append(t_ball)
    return R_balls
