import numpy as np
from sklearn.cross_decomposition import CCA
from utils.Normalization import maxmin
import torch
import math

def CCA_transform(feature_s, feature_t, n_components):
    """ CCA: Canonical Correlation Analysis
    """
    feature_s, feature_t = feature_s.cpu().detach().numpy(), feature_t.cpu().detach().numpy()
    cca = CCA(n_components=n_components, max_iter=10000)
    cca.fit(feature_s, feature_t)
    
    feature_s, feature_t = cca.transform(feature_s, feature_t)
    # feature_s = cca.transform(feature_s)
    # feature_t = cca.transform(feature_t)
    feature_s, feature_t = torch.tensor(feature_s).cuda(), torch.tensor(feature_t).cuda()
    return feature_s, feature_t

def CCA_writ(x,y,k):
    # 计算相关系数矩阵
    #   a[~(a==0).all(1)] #行是1列是0 删除全零行
    x, y = x.cpu().detach().numpy(),y.cpu().detach().numpy()
    x, y = maxmin(x), maxmin(y)
    matrix = np.vstack((x, y))
    R = np.cov(matrix)
    n = int(((R[0].shape)[0]))
    if n % 2 == 0:
        n = int(n/2)
        R11, R12, R21, R22 = R[:n, :n], R[:n,n:], R[n:,:n], R[n:,n:]
    else:
        n = int(n/2)
        R11, R12, R21, R22 = R[:n, :n], R[:n,n+1:], R[n+1:,:n], R[n+1:,n+1:]
    
    M1 = np.dot(np.dot(np.dot((np.linalg.pinv(R11)),R12),np.linalg.pinv(R22)),R21)
    M2 = np.dot(np.dot(np.dot((np.linalg.pinv(R22)),R21),np.linalg.pinv(R11)),R12)
    nx_features = (M1[0].shape)[0]
    ny_features = (M2[0].shape)[0]
    #使用函数np.linalg.eig()计算特征值和特征向量
    eig_val1, eig_vec1 = np.linalg.eig(M1)
    eig_val2, eig_vec2 = np.linalg.eig(M2)

    eig_pairs1 = [(np.abs(eig_val1[i]), eig_vec1[:,i]) for i in range(nx_features)]
    # sort eig_vec based on eig_val from highest to lowest
    eig_pairs1.sort(reverse=True, key=lambda x:x[0])
    # select the top k eig_vec
    feature1 = np.array([ele[1] for ele in eig_pairs1[:k]])

    inty = np.array([math.sqrt(ele[0])*ele[1] for ele in eig_pairs1[:k]])
    h = np.dot(np.dot((np.linalg.pinv(R11)),R12),np.transpose(inty))

    eig_pairs2 = [(np.abs(eig_val2[i]), eig_vec2[:,i]) for i in range(ny_features)]
    # sort eig_vec based on eig_val from highest to lowest
    eig_pairs2.sort(reverse=True, key=lambda x:x[0])
    # select the top k eig_vec
    feature2 = np.array([ele[1] for ele in eig_pairs2[:k]])

    feature1, feature2 = np.transpose(feature1),np.transpose(feature2)
    feature1, feature2 = torch.tensor(feature2).cuda(),torch.tensor(feature2).cuda()
    return feature1, feature2
  