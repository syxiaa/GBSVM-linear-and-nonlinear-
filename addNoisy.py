import pandas as pd
import collections
import copy
import warnings
import numpy as np
import csv
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")

def recreat_data(data,pre):
    """
    :return:
    """
    """
    df=pd.read_csv(url,header=None)
    data=df.values
    """
    numSamples,numAttribute=data.shape
    samNums=(int)(numSamples*pre)#需要置换标签总的样本数量
    df=pd.DataFrame(data)
    k_list=list(collections.Counter(data[:,numAttribute-1]).keys())
    v_list=list(collections.Counter(data[:,numAttribute-1]).values())
    # print(k_list)
    dff={}
    for i in range(len(k_list)):
        tag=k_list[i]#标签类别
        dff[i]=df[df[numAttribute-1]==int(tag)].reset_index()
        samNumsi=int(samNums*(v_list[i]/numSamples))#该类别中需要置换的样本数目
        temp_k=copy.deepcopy(k_list)
        temp_v=copy.deepcopy(v_list)
        temp_k.pop(i)
        temp_v.pop(i)
        k=0
        for j in range(len(temp_k)):
            samNumsij=int(samNumsi*(temp_v[j]/(sum(temp_v))))
            # print("sam",samNumsij)
            for l in range(k,samNumsij):
                dff[i].loc[l, numAttribute-1] = int(temp_k[j])
                k+=1
    new=dff[0]
    for i in range(len(k_list)-1):
        new=pd.concat([new,dff[i+1]])
    new = shuffle(new).reset_index().drop(['index', 'level_0'], axis=1)
    return new.values
    # new.to_csv(url1, index=False, header=None)
# pre=0.1
# data=[[1,2,1],
# [1,2,1],
# [1,2,1],
# [1,2,1],
#       [2,1,-1]]
# data=np.array(data)
# new=recreat_data(data,0.6)
# print(new)
# url1="D:\\py\粒球SVM_精度\划分后数据\\sonartrain.csv"
# url2="D:\py\粒球SVM_精度\噪声数据集\sonartrainN"+str(pre)+".csv"
# recreat_data(url1,url2,pre)

