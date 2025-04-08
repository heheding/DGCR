import numpy as np
import torch
'地址：https://github.com/heucoder/dimensionality_reduction_alo_codes/blob/master/codes/LPP/LPP.py'
def rbf(dist, t = 1.0):
    '''
    rbf kernel function
    '''
    return torch.exp(-(dist/t))
 
def cal_pairwise_dist(x):
 
    '''计算pairwise 距离, x是matrix
    (a-b)^2 = a^2 + b^2 - 2*a*b
    '''
    sum_x = torch.sum(torch.square(x), 1)
    dist = torch.add(torch.add(-2 * torch.mm(x, x.T), sum_x).T, sum_x) #二维及以下torch.T()，以上使用torch.transpose()
    #返回任意两个点之间距离的平方
    return dist
 
def cal_rbf_dist(data, n_neighbors = 10, t = 1):
 
    dist = cal_pairwise_dist(data)
    n = dist.shape[0]
    rbf_dist = rbf(dist, t)
    W = torch.zeros(n, n).cuda()
    for i in range(n):
        index_ = torch.argsort(dist[i])[1:1 + n_neighbors]
        index_ = index_.cpu().numpy()
        for j in range(index_.shape[0]):
            W[i, index_[j]] = rbf_dist[i, index_[j]]
            W[index_[j], i] = rbf_dist[index_[j], i]

    return W
 
def torch_lpp(data,
        n_dims = 128,
        n_neighbors = 30, t = 1.0):
    '''
    :param data: (n_samples, n_features)
    :param n_dims: target dim
    :param n_neighbors: k nearest neighbors
    :param t: a param for rbf
    :return:
    '''
    # dist = cal_pairwise_dist(data)
    # t = 0.01 * torch.max(dist)
    N = data.shape[0]
    W = cal_rbf_dist(data, n_neighbors, t).double()  #建立邻接矩阵W，参数有最近k个邻接点，以及热核参数t
    D = torch.zeros_like(W).double()
 
    for i in range(N):
        D[i,i] = torch.sum(W[i]) #求和每一行的元素的值，作为对角矩阵D的对角元素
 
    L = D - W  #L矩阵
    data = data.double()
    XDXT = torch.mm(torch.mm(torch.einsum('ij->ji', data), D), data)
    XLXT = torch.mm(torch.mm(torch.einsum('ij->ji', data), L), data)
 
    '''
    np.linalg.eig用作计算方阵的特征值以及右特征向量
    np.linalg.pinv对矩阵进行求逆
    本人理解：在求解最小特征问题（公式3）时，左乘XDX^T的逆，而后求其特征值以及对应特征向量
    输出的eig_val,eig_vec为特征值集合以及特征向量集合（此时是无序的）
    '''
    eig_val, eig_vec = torch.linalg.eig(torch.mm(torch.linalg.pinv(XDXT), XLXT))
    eig_val = eig_val.double() 
   
    '''
    argsort返回的是特征值在数据eig_val排序后的序号。
    如数组a=[2,1,7,4],则np.argsort(a)返回的是[1,0,3,2]
    '''
    sort_index_ = torch.argsort(torch.abs(eig_val))
    eig_val = eig_val[sort_index_]#对特征值进行排序
 
 
    #此时eig_val已经是升序排列了，需要排除前几个特征值接近0的数，至于为啥请看论文 "Locality Preserving Projections"
    j = 0
    while eig_val[j] < 1e-6: 
        j+=1
 
    #返回需要提取的前n个特征值所对应的n个特征向量的序列号
    sort_index_ = sort_index_[j:j+n_dims]
    # print(sort_index_)
    eig_val_picked = eig_val[j:j+n_dims]
    eig_vec_picked = eig_vec[:, sort_index_] #获取该n个特征向量组成的映射矩阵
 
    # data_ndim = np.dot(data, eig_vec_picked) #公式（4）
 
    return eig_vec_picked
