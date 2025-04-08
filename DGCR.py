from copy import deepcopy
import random
import math
from arg import get_args
from tokenize import Double
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
import pandas as pd
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from models.build_gen import VDI
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
import numpy as np
from matplotlib import pyplot as plt
from dataset.dataset_read import dataset_read
from loss import *
from models.loss import *
from utils.utils import *
import seaborn as sns
import datetime

seed_all(3407)
    
args = get_args()

de = nn.Linear(args.uz, 1)
decon1 = nn.Linear(args.hidden, 1)

def reconstruction_loss(recon, input, name):
    if name == "L1":
        rec_loss = nn.L1Loss()
    elif name == "MSE":
        rec_loss = nn.MSELoss()
    else:
        rec_loss = nn.L1Loss()

    return rec_loss(recon, input)

def KLD_loss(mu0, log_var0, mu1, log_var1):
    kld_loss = torch.sum(-0.5 * (1 + (log_var0 - log_var1) - ((mu0 - mu1) ** 2 + log_var0.exp() / log_var1.exp())), dim=(1, 2))
    return kld_loss

def KLD_loss1(mu, log_var):
    # 计算每个样本的 KLD 损失
    kld_loss = -0.5 * (1 + log_var - mu ** 2 - log_var.exp())
    # 对每个样本的损失进行求和，结果维度为 [6]
    kld_loss = torch.mean(torch.sum(kld_loss, dim=1), dim=1)
    return kld_loss

def compute_loss_recon(x_seq, r_x):
    # 计算逐元素的 MSELoss，并保持所有维度
    mse_loss = F.mse_loss(x_seq, r_x, reduction='none')  # 保留所有维度的损失
    
    # 对除了第一个维度（6）以外的维度求平均
    loss_recon = torch.mean(mse_loss, dim=(1, 2, 3))  # 在 (256, 30, 13) 上求平均，保留 6
    return loss_recon

def normalize_and_process(row, first_elements_end, other_elements_start):
    """
    对指定的行进行归一化处理，并返回归一化后的处理结果。
    row: 输入行数据
    first_elements_end: 选取第一个元素的结束位置（索引）
    other_elements_start: 选取其他元素的开始位置（索引）
    """
    first_elements = row[:first_elements_end]
    other_elements = row[other_elements_start:]
    
    total_sum = first_elements.sum() + other_elements.sum()
    
    # 归一化
    processed_elements = torch.cat((first_elements / total_sum, other_elements / total_sum))
    return processed_elements.unsqueeze(0)

def plot_inv_spf(u_inv, u_spf):
    inv = decon1(u_inv).squeeze(-1)
    inv = inv[:,0:100].cpu().detach().numpy()
    spf = de(u_spf).squeeze(-1)
    spf = spf[:,0:100].cpu().detach().numpy()
    
    r = len(inv[0]) + 1
    
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rc('font',family='Times New Roman')
    plt.figure(dpi=300, figsize=(14, 8))
    
    # plt.ylim(-0.055, 0.035)
    
    plt.plot(np.arange(1, r), inv[0], 'r-', label=r"$\beta^{din}_{S_{1}}$")
    plt.plot(np.arange(1, r), inv[1], 'g-', label=r"$\beta^{din}_{S_{2}}$")
    plt.plot(np.arange(1, r), inv[2], 'b-', label=r"$\beta^{din}_{S_{3}}$")
    plt.plot(np.arange(1, r), inv[3], 'm-', label=r"$\beta^{din}_{S_{4}}$")
    plt.plot(np.arange(1, r), inv[4], 'b-', label=r"$\beta^{din}_{S_{5}}$")
    plt.plot(np.arange(1, r), inv[5], 'm-', label=r"$\beta^{din}_{S_{6}}$")
    plt.xlabel('Sample', size=28)
    plt.ylabel('Output', size=28)
    plt.tick_params(labelsize=24)
    plt.legend(fontsize=22)
    plt.tight_layout()
    plt.savefig(f"/root/dh/2025/OODjuzhi/figs/out/inv.png")
    plt.savefig(f"/root/dh/2025/OODjuzhi/figs/out/inv.eps")
    plt.close()

    r = len(spf[0]) + 1
    
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rc('font',family='Times New Roman')
    plt.figure(dpi=300, figsize=(14, 8))
    # plt.ylim(-0.010, 0.030)
    plt.plot(np.arange(1, r), spf[0], 'r-', label=r"$\beta^{dva}_{S_{1}}$")
    plt.plot(np.arange(1, r), spf[1], 'g-', label=r"$\beta^{dva}_{S_{2}}$")
    plt.plot(np.arange(1, r), spf[2], 'b-', label=r"$\beta^{dva}_{S_{3}}$")
    plt.plot(np.arange(1, r), spf[3], 'm-', label=r"$\beta^{dva}_{S_{4}}$")
    plt.plot(np.arange(1, r), spf[4], 'b-', label=r"$\beta^{dva}_{S_{5}}$")
    plt.plot(np.arange(1, r), spf[5], 'm-', label=r"$\beta^{dva}_{S_{6}}$")
    plt.xlabel('Sample', size=28)
    plt.ylabel('Output', size=28)
    plt.tick_params(labelsize=24)
    plt.legend(fontsize=22)
    plt.tight_layout()
    plt.savefig(f"/root/dh/2025/OODjuzhi/figs/out/spf.png")
    plt.savefig(f"/root/dh/2025/OODjuzhi/figs/out/spf.eps")
    plt.close()
    


def calculate_metrics(output, label):
    
    MAE = mean_absolute_error(output, label)
    RMSE = sqrt(mean_squared_error(output, label))
    R2 = r2_score(output, label)
    return MAE, RMSE, R2

def calculate_output(test1, test2, label_test,normal_mu, normal_log_var):
    # 第一列为测试域
    uin, uin_mu, uin_log_var = UNin(test1)   # u, u_mu, u_log_var 6*256*10
    uva, uva_mu, uva_log_var = UNva(test1)   # u, u_mu, u_log_var 6*256*10
    uno, uno_mu, uno_log_var = UNno(test1)
    q_z1, q_mu1, q_log_var1 = Q_ZNet(uin, test1)
    r_x = ReconstructNet(uin, uva)
    y_seq0 = PredNet1(q_z1[0])
    y_seq1 = PredNet2(q_z1[1])
    y_seq2 = PredNet3(q_z1[2])
    y_seq3 = PredNet4(q_z1[3])
    
    inkl = KLD_loss(uin_mu, uin_log_var, normal_mu, normal_log_var)
    vakl = KLD_loss(uva_mu, uva_log_var, normal_mu, normal_log_var)
    
    # reconstruction loss (p(x|u))
    loss_recon = compute_loss_recon(test1, r_x)
    
    # I(in; va)
    infor = loss_recon + vakl + inkl
    iny = torch.sum(infor)
    inforg = infor/iny
    
    js01 = KLD_loss(q_mu1[0].unsqueeze(0), q_log_var1[0].unsqueeze(0), q_mu1[1].unsqueeze(0), q_log_var1[1].unsqueeze(0))
    js02 = KLD_loss(q_mu1[0].unsqueeze(0), q_log_var1[0].unsqueeze(0), q_mu1[2].unsqueeze(0), q_log_var1[2].unsqueeze(0))
    js03 = KLD_loss(q_mu1[0].unsqueeze(0), q_log_var1[0].unsqueeze(0), q_mu1[3].unsqueeze(0), q_log_var1[3].unsqueeze(0))

    # 第一列为测试域
    uin, uin_mu, uin_log_var = UNin(test2)   # u, u_mu, u_log_var 6*256*10
    uva, uva_mu, uva_log_var = UNva(test2)   # u, u_mu, u_log_var 6*256*10
    uno, uno_mu, uno_log_var = UNno(test2)
    q_z, q_mu, q_log_var = Q_ZNet(uin,test2)
    r_x = ReconstructNet(uin, uva)
    z_x = Z_ReconstructNet(q_z)
    y_seq00 = PredNet1(q_z[0])
    y_seq1 = PredNet2(q_z[1])
    y_seq2 = PredNet3(q_z[2])
    y_seq3 = PredNet4(q_z[3])
    
    js201 = KLD_loss(q_mu[0].unsqueeze(0), q_log_var[0].unsqueeze(0), q_mu[1].unsqueeze(0), q_log_var[1].unsqueeze(0))
    
    sumz = 1/js201+1/js01+1/js02+1/js03
    js201,js01,js02,js03 = 1/js201/sumz,1/js01/sumz,1/js02/sumz,1/js03/sumz
    
    yt = js201*y_seq0+js01*y_seq1+js02*y_seq2+js03*y_seq3
    
    label = label_test

    output = (yt*test_std + test_mean).detach().cpu().numpy()
    label = (label*test_std + test_mean).detach().cpu().numpy()
    return output, label

device = torch.device("cuda:0")
batch_size = args.batch_size
lr = args.lr   
n_epoch = args.n_epoch
interval = 20 
weight_decay = 0
lambda_pre = args.lambda_pre
lambda_rel = args.lambda_rel
lambda_inv = args.lambda_inv
lambda_infor = args.lambda_infor
normal_mu = torch.tensor(np.float32(0)).to(device)
normal_log_var = torch.tensor(np.float32(0)).to(device)
domain = 4
# 初始化存储KL散度的矩阵
js_matrix = torch.zeros(domain, domain).to(device)
savemodel = False
loadmodel = False

s1 = pd.read_csv('/root/dh/2024/OODCo/data/2011.csv')
s2 = pd.read_csv('/root/dh/2024/OODCo/data/2012.csv')
s3 = pd.read_csv('/root/dh/2024/OODCo/data/2013.csv')
s4 = pd.read_csv('/root/dh/2024/OODCo/data/2014.csv')
tdata = pd.read_csv('/root/dh/2024/OODCo/data/2015.csv')

test_mean, test_std, dataset1,dataset2,dataset3,dataset4, test, label_test,\
        test_trn1, y_test_trn1, test_trn2, y_test_trn2, test_trn3, y_test_trn3, \
        test_trn4, y_test_trn4 = dataset_read(s1,s2,s3,s4,tdata, batch_size = batch_size)

z = 1/domain

test1 = torch.stack([test, test_trn2, test_trn3, test_trn4], dim=0)
label_test1 = torch.stack([label_test, y_test_trn2, y_test_trn3, y_test_trn4], dim=0)
test2 = torch.stack([test_trn1, test, test, test], dim=0)
label_test2 = torch.stack([label_test, y_test_trn1, label_test, label_test], dim=0)
test1 = test1.cuda().to(torch.float32)
label_test1 = label_test1.cuda().to(torch.float32)
test2 = test2.cuda().to(torch.float32)
label_test2 = label_test2.cuda().to(torch.float32)
label_test = label_test.cuda().to(torch.float32)
test_mean, test_std = test_mean[-2],test_std[-2]

x_seq = []
d_seq = []

z_seq = to_tensor(np.zeros((domain, batch_size, 1), dtype=np.float32)) + z
# lr=1e-4 batch_size=64
cuda = True
cudnn.benchmark = True

model_path = '/root/dh/2025/OODCO/model_path/lilun'
de = de.to(device)
decon1 = decon1.to(device)
UNin = VDI('UNin').to(device)
UNva = VDI('UNva').to(device)
UNno = VDI('UNno').to(device)
PredNet1 = VDI('PredNet1').to(device)
PredNet2 = VDI('PredNet2').to(device)
PredNet3 = VDI('PredNet3').to(device)
PredNet4 = VDI('PredNet4').to(device)
Q_ZNet = VDI('Q_ZNet').to(device)
ReconstructNet = VDI('ReconstructNet').to(device)
Z_ReconstructNet = VDI('ZReconstructNet').to(device)
SAR = VDI('sar').to(device)

if loadmodel:
    UNin.load_state_dict(torch.load(model_path+'/UNin.pth'))
    UNva.load_state_dict(torch.load(model_path+'/UNva.pth'))
    Q_ZNet.load_state_dict(torch.load(model_path+'/Q_ZNet.pth'))
    ReconstructNet.load_state_dict(torch.load(model_path+'/ReconstructNet.pth'))
    SAR.load_state_dict(torch.load(model_path+'/SAR.pth'))
    PredNet1.load_state_dict(torch.load(model_path+'/PredNet1.pth'))
    PredNet2.load_state_dict(torch.load(model_path+'/PredNet2.pth'))
    PredNet3.load_state_dict(torch.load(model_path+'/PredNet3.pth'))
    PredNet4.load_state_dict(torch.load(model_path+'/PredNet4.pth'))
    
crossentropyloss=nn.CrossEntropyLoss()
loss_predict = torch.nn.MSELoss(reduction='mean')

UZF_parameters = list(UNin.parameters()) + list(Q_ZNet.parameters()) + list(
                    Z_ReconstructNet.parameters()) + list(UNva.parameters()) + list(
                    UNno.parameters()) + list(PredNet1.parameters()) + list(
                    ReconstructNet.parameters()) + list(SAR.parameters())+ list(
                    PredNet2.parameters()) + list(PredNet3.parameters()) + list(
                    PredNet4.parameters())

optimizer_UZF = optim.Adam(UZF_parameters, lr=lr, weight_decay=weight_decay)

for epoch in range(n_epoch):
    for batch_idx, ((x_seq0, y_seq0), (x_seq1, y_seq1),(x_seq2, y_seq2),(x_seq3, y_seq3)) in enumerate(zip(dataset1, dataset2, dataset3, dataset4)):
        
        x_seq = torch.stack([x_seq0.cuda(), x_seq1.cuda(), x_seq2.cuda(), x_seq3.cuda()], dim=0)
        y_lable = torch.stack([y_seq0.cuda(), y_seq1.cuda(), y_seq2.cuda(), y_seq3.cuda()], dim=0)
        x_seq = x_seq.to(torch.float32)
        y_lable = y_lable.to(torch.float32)

        optimizer_UZF.zero_grad()
        
        uin, uin_mu, uin_log_var = UNin(x_seq)   # u, u_mu, u_log_var 6*256*10
        uva, uva_mu, uva_log_var = UNva(x_seq)   # u, u_mu, u_log_var 6*256*10
        uno, uno_mu, uno_log_var = UNno(x_seq)
        q_z, q_mu, q_log_var = Q_ZNet(uin, x_seq)
        r_x = ReconstructNet(uin, uva)
        y_seq0 = PredNet1(q_z[0])
        y_seq1 = PredNet2(q_z[1])
        y_seq2 = PredNet3(q_z[2])
        y_seq3 = PredNet4(q_z[3])
        d_inv = SAR(uin)
        
        loss_inv1 = F.kl_div(d_inv.softmax(dim=1).log(), z_seq.softmax(dim=1), reduction='sum')
        loss_inv2 = F.kl_div(z_seq.softmax(dim=1).log(), d_inv.softmax(dim=1), reduction='sum')
        # loss_inv = crossentropyloss(d_inv, z_seq)
        loss_inv = (loss_inv1+loss_inv2)/2
        
        inkl = KLD_loss(uin_mu, uin_log_var, normal_mu, normal_log_var)
        vakl = KLD_loss(uva_mu, uva_log_var, normal_mu, normal_log_var)
        
        # reconstruction loss (p(x|u))
        loss_recon = compute_loss_recon(x_seq, r_x)
        
        # I(in; va)
        infor = loss_recon + vakl + inkl
        loss_infor = torch.mean(infor)
        iny = torch.sum(infor)
        inforg = infor/iny
        
        # similarity 计算不同域之间的不变相似度，而不是同一域的
        
        js01 = KLD_loss(uin_mu[0].unsqueeze(0), uin_log_var[0].unsqueeze(0), uin_mu[1].unsqueeze(0), uin_log_var[1].unsqueeze(0))
        js02 = KLD_loss(uin_mu[0].unsqueeze(0), uin_log_var[0].unsqueeze(0), uin_mu[2].unsqueeze(0), uin_log_var[2].unsqueeze(0))
        js03 = KLD_loss(uin_mu[0].unsqueeze(0), uin_log_var[0].unsqueeze(0), uin_mu[3].unsqueeze(0), uin_log_var[3].unsqueeze(0))
        js12 = KLD_loss(uin_mu[1].unsqueeze(0), uin_log_var[1].unsqueeze(0), uin_mu[2].unsqueeze(0), uin_log_var[2].unsqueeze(0))
        js13 = KLD_loss(uin_mu[1].unsqueeze(0), uin_log_var[1].unsqueeze(0), uin_mu[3].unsqueeze(0), uin_log_var[3].unsqueeze(0))
        js23 = KLD_loss(uin_mu[2].unsqueeze(0), uin_log_var[2].unsqueeze(0), uin_mu[3].unsqueeze(0), uin_log_var[3].unsqueeze(0))

        sum0 = 1/js01+1/js02+1/js03+1/inforg[0]
        sum1 = 1/js01+1/js12+1/js13+1/inforg[1]
        sum2 = 1/js02+1/js12+1/js23+1/inforg[2]
        sum3 = 1/js03+1/js13+1/js23+1/inforg[3]
        
        js0, js001, js002, js003 = 1/inforg[0]/sum0, 1/js01/sum0, 1/js02/sum0, 1/js03/sum0
        js1, js101, js112, js113 = 1/inforg[1]/sum1, 1/js01/sum1, 1/js12/sum1, 1/js13/sum1
        js2, js202, js212, js223 = 1/inforg[2]/sum2, js02/sum2, 1/js12/sum2, 1/js23/sum2
        js3, js303, js313, js323 = 1/inforg[3]/sum3, 1/js03/sum3, 1/js13/sum3, 1/js23/sum3

        
        y0 = js0 * y_seq0 + js001 * y_seq1 + js002 * y_seq2 + js003 * y_seq3
        y1 = js1 * y_seq1 + js101 * y_seq0 + js112 * y_seq2 + js113 * y_seq3       
        y2 = js2 * y_seq2 + js202 * y_seq0 + js212 * y_seq1 + js223 * y_seq3       
        y3 = js3 * y_seq3 + js303 * y_seq0 + js313 * y_seq1 + js323 * y_seq2 

        y_rel = torch.stack([y0, y1, y2, y3], dim=0)
        y_seq = torch.stack([y_seq0, y_seq1, y_seq2, y_seq3], dim=0)
        
        loss_rel = loss_predict(flat(y_rel), flat(y_lable))
        # E_q[log p(y|z)]
        loss_p_y_z = loss_predict(flat(y_seq), flat(y_lable))

        loss_E = lambda_pre*loss_p_y_z + lambda_rel * loss_rel + lambda_inv * loss_inv + lambda_infor * loss_infor
        loss_E.backward()

        optimizer_UZF.step()
    with torch.no_grad():    
        if epoch % interval == 0 and epoch != 0:   # 第interval轮全部训练完
            print('Train Epoch: {}\t  loss_E: {:.6f}\t loss_pre: {:.6f}\t loss_rel: {:.6f}\t loss_inv: {:.6f}\t loss_infor: {:.6f}\t'.format(
                epoch, loss_E.data, loss_p_y_z.data, loss_rel.data, loss_inv.data, loss_infor.data))  # 打印源域预测损失和总的损失
            
            output, label = calculate_output(test1, test2, label_test,normal_mu, normal_log_var)

            MAE, RMSE, R2 = calculate_metrics(output, label)
            print('MAE: {} \t RMSE: {} \t  R2: {} \t '.format(MAE, RMSE, R2))
            correlation_matrix = np.abs(np.corrcoef(uin.flatten().cpu().detach().numpy(), uva.flatten().cpu().detach().numpy()))
            cor = correlation_matrix[0, 1]
            # plot_inv_spf(q_z, uva)
            a4 = args.xulie
            numbers = [a4, epoch, MAE, RMSE, R2]
            with open('/root/dh/2025/OODCO/output.txt', 'a') as f:
                # 将浮点数列表转换为字符串，并写入文件的第一行
                f.write(' '.join(map(str, numbers)) + '\n')
       
if savemodel:
    ensure_path(model_path)
    torch.save(UNin.state_dict(), model_path+'/UNin.pth')
    torch.save(UNva.state_dict(), model_path+'/UNva.pth')
    torch.save(ReconstructNet.state_dict(), model_path+'/ReconstructNet.pth')
    torch.save(Q_ZNet.state_dict(), model_path+'/Q_ZNet.pth')
    torch.save(SAR.state_dict(), model_path+'/SAR.pth') 
    torch.save(PredNet1.state_dict(), model_path+'/PredNet1.pth')
    torch.save(PredNet2.state_dict(), model_path+'/PredNet2.pth') 
    torch.save(PredNet3.state_dict(), model_path+'/PredNet3.pth') 
    torch.save(PredNet4.state_dict(), model_path+'/PredNet4.pth')   
     
# test
with torch.no_grad():
    output, label = calculate_output(test1, test2, label_test,normal_mu, normal_log_var)

# np.savetxt('/root/dh/2025/OODjuzhi/out/Label.csv', label, delimiter=',')
np.savetxt('/root/dh/2025/OODCO/out/DGCR.csv', output, delimiter=',') 
    
MAE, RMSE, R2 = calculate_metrics(output, label)
print('TEST MAE: {} \t RMSE: {} \t  R2: {} \t '.format(MAE, RMSE, R2))


r = output.shape[0] + 1
plt.figure(dpi=300,figsize=(28,7))
plt.rc('font', family='Times New Roman')
plt.grid(True)
# 设置网格线格式：
plt.grid(color='gray',    
        linestyle='-.',
        linewidth=1,
        alpha=0.3) 
plt.plot(np.arange(1, r), label, color='blue', linestyle='-',lw=1.5, label='Real')
plt.plot(np.arange(1, r), output, color='red', linestyle='-', label='IDTL')
plt.tick_params(labelsize=24)
plt.xlabel('Sample Point',fontsize=28)
plt.ylabel('Oligomer Density (kg/m3)', size=28)
plt.legend(fontsize=22)
plt.savefig(f"/root/dh/2025/OODCO/figs/DGCR.png")
plt.close()
current_time = datetime.datetime.now()
print(f"VDISDA聚酯,当前时间：{current_time}, Batch size: {batch_size}, 学习率: {lr}, epoch: {n_epoch}")
