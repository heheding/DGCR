
from sklearn import preprocessing
import matplotlib.pyplot as plt


#Z-Score标准化
def zscore(data):
    #建立StandardScaler对象
    zscore = preprocessing.StandardScaler()
    # 标准化处理
    data_zs = zscore.fit_transform(data)
    return data_zs

#Max-Min标准化
def maxmin(data):
    #建立MinMaxScaler对象
    minmax = preprocessing.MinMaxScaler()
    # 标准化处理
    data_minmax = minmax.fit_transform(data)
    return data_minmax

#MaxAbs标准化
def maxabs(data):
    #建立MinMaxScaler对象
    maxabs = preprocessing.MaxAbsScaler()
    # 标准化处理
    data_maxabs = maxabs.fit_transform(data)
    return data_maxabs

#RobustScaler标准化
def robustscaler(data):
    #建立RobustScaler对象
    robust = preprocessing.RobustScaler()
    # 标准化处理
    data_rob = robust.fit_transform(data)
    return data_rob
