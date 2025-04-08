import torch
from torch import nn
import random
# from args import args
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, cohen_kappa_score, confusion_matrix, accuracy_score
import os
import sys
import logging
# import wandb
from shutil import copy
### For tsne
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch.nn.init as init
import antropy as ant
import torch.nn.functional as F

def curves(df_label, df_cida, df_dgcr, df_sad, df_vdi, df_nu):
    
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rc('font',family='Times New Roman')
    plt.figure(dpi=300,figsize=(24, 7))
    # 画出误差曲线
    plt.grid(True)
    # 设置网格线格式：
    plt.grid(color='gray',    
            linestyle='-.',
            linewidth=1,
            alpha=0.3) 
    plt.plot(np.array(df_label),color = 'blue',linestyle='-', lw=2, label='Label')
    plt.plot(np.array(df_cida),color = 'c',lw=1,linestyle='dashed', label='CIDA')
    plt.plot(np.array(df_sad),color = 'goldenrod',lw=1,linestyle='dashed', label='SAD')
    plt.plot(np.array(df_vdi),color = 'k',lw=1,linestyle='dashed', label='VDI')
    plt.plot(np.array(df_nu),color = 'y',lw=1,linestyle='dashed',label='NU')
    plt.plot(np.array(df_dgcr),color = 'r',lw=2,linestyle='-',label='DGCR')
        
    plt.tick_params(labelsize=24)
    plt.xlabel('Sample',fontsize=28)
    plt.ylabel('CO Values (mg/m$^3$)', size=28)
    plt.legend(fontsize=22)
    plt.tight_layout()
    plt.savefig(f"/root/dh/2025/OODCO/figs/curves.png")
    plt.savefig(f"/root/dh/2025/OODCO/figs/curves.eps")

######1
#带空心圆和叉叉
def curves2(df_label, df_adarnn):
    plt.figure(dpi=300, figsize=(24, 7))
    plt.rc('font', family='Times New Roman')
    # 画出误差曲线
    plt.grid(True)
    # 设置网格线格式：
    plt.grid(color='gray',    
            linestyle='-.',
            linewidth=1,
            alpha=0.3) 
    plt.plot(df_label, color='blue', linestyle='-',lw=1.5, label='Real')
    plt.plot(df_adarnn, color='red', linestyle='-', label='DGCR')
    plt.tick_params(labelsize=24)
    plt.xlabel('Sample',fontsize=28)
    plt.ylabel('CO Values (mg/m$^3$)', size=28)
    plt.legend(fontsize=22)
    plt.tight_layout()
    plt.savefig(f"/root/dh/2025/OODCO/figs/DGCR.png")
    plt.savefig(f"/root/dh/2025/OODCO/figs/DGCR.eps")

def errplt(df_label, df_cida, df_dgcr, df_sad, df_vdi, df_nu):
    
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rc('font',family='Times New Roman')
    plt.figure(dpi=300,figsize=(24, 7))
    # 画出误差曲线
    plt.grid(True)
    # 设置网格线格式：
    plt.grid(color='gray',    
            linestyle='-.',
            linewidth=1,
            alpha=0.3) 
    plt.plot(np.array(df_cida - df_label),'c-v',lw=1,ls='dashed', label='CIDA')
    plt.plot(np.array(df_sad - df_label),'g-*',lw=1,ls='dashed', label='SAD')
    plt.plot(np.array(df_vdi - df_label),'b-+',lw=1,ls='dashed', label='VDI')
    plt.plot(np.array(df_nu - df_label),color = 'goldenrod',marker='^',lw=1,ls='dashed', label='NU')
    plt.plot(np.array(df_dgcr - df_label),'r-o',lw=1.5,label='DGCR')
    
        
    plt.axhline(y=0, color='k', ls='--')  # 添加参考线表示误差为0
    # plt.ylim(-0.6, 0.4)
    plt.tick_params(labelsize=24)
    plt.xlabel('Sample',fontsize=28)
    plt.ylabel('Prediction Error',fontsize=28)
    plt.legend(fontsize=22)
    plt.tight_layout()
    plt.savefig(f"/root/dh/2025/OODCO/figs/errplt.png")   
    plt.savefig(f"/root/dh/2025/OODCO/figs/errplt.eps")

##############3
def scatter(df_label, df_cida, df_dgcr, df_sad, df_vdi, df_nu):
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rc('font',family='Times New Roman')
    plt.figure(dpi=300,figsize=(14, 8))
    c=np.arange(-5,9)
    plt.plot(c,c,color="blue",linewidth=2, linestyle="-",label='Real', zorder=0)
    plt.scatter(df_label, df_cida,marker='^',color = 'g',s=80,label='CIDA', zorder=2)
    plt.scatter(df_label, df_sad,marker='o',color = 'c',s=80,label='SAD', zorder=3)
    plt.scatter(df_label, df_vdi,marker='>',color = 'y',s=80,label='VDI', zorder=4)
    plt.scatter(df_label, df_nu,marker='+',color = 'k',s=80,label='NU', zorder=5)
    plt.scatter(df_label, df_dgcr,marker='*',color = 'r',s=90,label='IDTL', zorder=6)
    plt.xlabel('Real Values',size=28)
    plt.ylabel('Predicted Values',size=28)
    plt.tick_params(labelsize=24)
    plt.legend(fontsize=22)
    plt.tight_layout()
    plt.savefig(f"/root/dh/2025/OODCO/figs/scatter.png")
    plt.savefig(f"/root/dh/2025/OODCO/figs/scatter.eps")
    
def plot_boxplot(df_label, df_cida, df_dgcr, df_sad, df_vdi, df_nu):
    labels = ['CIDA', 'SAD', 'VDI', 'NU', 'DGCR']
    data = [abs(df_cida - df_label), abs(df_sad - df_label), abs(df_vdi - df_label), abs(df_nu - df_label), abs(df_dgcr - df_label)]
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rc('font',family='Times New Roman')
    fig = plt.figure(dpi=300, figsize=(14, 8))
    
    plt.grid(True)  # 显示网格
    plt.boxplot(data,
            medianprops={'color': 'red', 'linewidth': '1.5'},
            meanline=True,
            showmeans=True,
            meanprops={'color': 'blue', 'ls': '--', 'linewidth': '1.5'},
            flierprops={"marker": "o", "markerfacecolor": "red", "markersize": 10},
            labels=labels)
    plt.tight_layout()
    plt.xlabel('Methods', size=28)
    plt.ylabel('Absolute value of predicted error', size=28)
    plt.tick_params(labelsize=24)

    plt.savefig("/root/dh/2025/OODCO/figs/boxplot.png")
    plt.savefig("/root/dh/2025/OODCO/figs/boxplot.eps")

############################################################################
def copy_Files(destination, data_type, da_method):
    destination_dir = os.path.join(destination, "MODEL_BACKUP_FILES")
    os.makedirs(destination_dir, exist_ok=True)
    copy("train_CD.py", os.path.join(destination_dir, "train_CD.py"))
    copy(f"trainer/{da_method}.py", os.path.join(destination_dir, f"{da_method}.py"))
    copy(f"trainer/training_evaluation.py", os.path.join(destination_dir, f"training_evaluation.py"))
    copy(f"config_files/{data_type}_Configs.py", os.path.join(destination_dir, f"{data_type}_Configs.py"))
    copy("dataloader/dataloader.py", os.path.join(destination_dir, "dataloader.py"))
    copy(f"models/models.py", os.path.join(destination_dir, f"models.py"))
    copy("args.py",  os.path.join(destination_dir, f"args.py"))

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def ensure_path(path):
    if os.path.exists(path):
        pass
    else:
        os.makedirs(path)

def seed_all(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)

###

def set_requires_grad(model, requires_grad=True):
    for param in model.parameters():
        param.requires_grad = requires_grad


def loop_iterable(iterable):
    while True:
        yield from iterable


def fix_randomness(SEED):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark  = False

def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)
        # if name=='weight':
        #     nn.init.kaiming_uniform_(param.data)
        # else:
        #     torch.nn.init.zeros_(param.data)


############ FOR Domain_Mixup #################
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.1)
        m.bias.data.fill_(0)

def exp_lr_scheduler(optimizer, init_lr, lrd, nevals):
    """Implements torch learning reate decay with SGD"""
    lr = init_lr / (1 + nevals*lrd)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer
#################################################

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def mean_std(x):
    mean = np.mean(np.array(x))
    std = np.std(np.array(x))
    return mean, std


def save_to_df_1(run_id, data_id, scores):
    res = []
    for metric in scores:
        mean = np.mean(np.array(metric))
        std = np.std(np.array(metric))
        res.append(f'{mean:2.2f}')
        res.append(f'{std:2.2f}')
    df_out = pd.Series((run_id, data_id, res))
    # df_out = pd.Series((run_id, data_id, res[0][0],res[0][1],res[1][0],res[1][1],res[2][0],res[2][1],res[2][0],res[2][1],res[3][0],res[3][1]))
    return df_out


def save_to_df(scores):
    res = []
    for metric in scores:
        mean = np.mean(np.array(metric))
        std = np.std(np.array(metric))
        res.append(f'{mean:2.5f}')
        res.append(f'{std:2.5f}')
    # df_out = pd.Series((run_id, data_id, res[0][0],res[0][1],res[1][0],res[1][1],res[2][0],res[2][1],res[2][0],res[2][1],res[3][0],res[3][1]))
    return res

def report_results(df,data_type, da_method, exp_log_dir):

    printed_results = df[['src_id', 'tgt_id', 'Source_only_Acc_mean', f'{da_method}_Acc_mean']]
    printed_results.columns = ['src_id', 'tgt_id', 'Source_only', f'{da_method}']
    mean_src_only = pd.to_numeric(printed_results['Source_only']).mean()
    mean_da_method = pd.to_numeric(printed_results[f'{da_method}']).mean()
    printed_results.loc[len(printed_results)] = ['mean', 'mean', mean_src_only,mean_da_method]
    print_res_name = os.path.basename(exp_log_dir)
    df.to_excel(f'{exp_log_dir}/full_res_results_{print_res_name}.xlsx')
    printed_results.to_excel(f'{exp_log_dir}/printed_results_{print_res_name}.xlsx')
    return printed_results


def _logger(logger_name, level=logging.DEBUG):
    """
    Method to return a custom logger with the given name and level
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    # format_string = ("%(asctime)s — %(name)s — %(levelname)s — %(funcName)s:"
    #                 "%(lineno)d — %(message)s")
    format_string = "%(message)s"
    log_format = logging.Formatter(format_string)
    # Creating and adding the console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)
    # Creating and adding the file handler
    file_handler = logging.FileHandler(logger_name, mode='a')
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)
    return logger



def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    import io
    from PIL import Image

    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img


def _plot_tsne(model, src_dl, tgt_dl, device, save_dir, model_type,
               train_mode):  # , layer_output_to_plot, y_test, save_dir, type_id):
    print("Plotting TSNE for " + model_type + "...")

    with torch.no_grad():
        model = model.to('cpu')
        src_data = src_dl.dataset.x_data.float()
        src_labels = src_dl.dataset.y_data.view((-1)).long()  # .to(device)
        src_predictions, (src_features,_)= model(src_data)

        tgt_data = tgt_dl.dataset.x_data.float()
        tgt_labels = tgt_dl.dataset.y_data.view((-1)).long()  # .to(device)
        tgt_predictions, (tgt_features,_) = model(tgt_data)

    perplexity = 50
    src_model_tsne = TSNE(n_components=2, random_state=1, perplexity=perplexity).fit_transform(
        (Variable(src_features).data).detach().cpu().numpy().reshape(len(src_labels), -1).astype(np.float64))

    tgt_model_tsne = TSNE(n_components=2, random_state=1, perplexity=perplexity).fit_transform(
        (Variable(tgt_features).data).detach().cpu().numpy().reshape(len(tgt_labels), -1).astype(np.float64))

    plt.figure(figsize=(16, 10))

    cmaps = plt.get_cmap('jet')
    src_scatter = plt.scatter(src_model_tsne[:, 0], src_model_tsne[:, 1], s=20, c=src_labels, cmap=cmaps,
                              label="source data")
    tgt_scatter = plt.scatter(tgt_model_tsne[:, 0], tgt_model_tsne[:, 1], s=20, c=tgt_labels, cmap=cmaps,
                              label="target data", marker='^')
    handles, _ = src_scatter.legend_elements(prop='colors')
    plt.legend(handles, tgt_labels.numpy(), loc="lower left", title="Classes")

    if not os.path.exists(os.path.join(save_dir, "tsne_plots")):
        os.mkdir(os.path.join(save_dir, "tsne_plots"))

    file_name = "tsne_" + model_type + "_" + train_mode + ".png"
    fig_save_name = os.path.join(save_dir, "tsne_plots", file_name)
    # wandb.log({f"{file_name}": wandb.Image(plt)})
    plt.savefig(fig_save_name)
    plt.close()

    plt.figure(figsize=(16, 10))
    plt.scatter(src_model_tsne[:, 0], src_model_tsne[:, 1], s=10, c='red',
                label="source data")
    plt.scatter(tgt_model_tsne[:, 0], tgt_model_tsne[:, 1], s=10, c='blue',
                label="target data")
    plt.legend()

    file_name = "tsne_" + model_type + "_" + train_mode + "_domain-based.png"
    fig_save_name = os.path.join(save_dir, "tsne_plots", file_name)
    plt.savefig(fig_save_name)
    # wandb.log({f"{file_name}": wandb.Image(plt)})
    plt.close()
    model = model.to(device)


def plot_tsne_one_domain(model, src_dl, device, save_dir, model_type, train_mode):
    with torch.no_grad():
        src_data = src_dl.dataset.x_data.float().to(device)
        src_labels = src_dl.dataset.y_data.view((-1)).long()  # .to(device)
        src_predictions, src_features = model(src_data)

    perplexity = 50
    src_model_tsne = TSNE(n_components=2, random_state=1, perplexity=perplexity).fit_transform(
        (Variable(src_features).data).detach().cpu().numpy().reshape(len(src_labels), -1).astype(np.float64))

    fig, ax = plt.subplots(figsize=(16, 10))
    cmaps = plt.get_cmap('jet')
    src_scatter = ax.scatter(src_model_tsne[:, 0], src_model_tsne[:, 1], s=20, c=src_labels, cmap=cmaps,
                             label="source data")

    legend1 = ax.legend(*src_scatter.legend_elements(),
                        loc="lower left", title="Classes")
    ax.add_artist(legend1)
    ax.legend()
    if not os.path.exists(os.path.join(save_dir, "tsne_plots")):
        os.mkdir(os.path.join(save_dir, "tsne_plots"))

    file_name = "tsne_" + model_type + "_" + train_mode + ".png"
    fig_save_name = os.path.join(save_dir, "tsne_plots", file_name)
    plt.savefig(fig_save_name)
    # wandb.log({"fig_save_name": fig})

    plt.close()

    # plotly
    # fig1=px.scatter(x=src_model_tsne[:, 0], y=src_model_tsne[:, 1], color=src_labels.numpy().astype(str), labels={'color': 'Classes'})
    # px.scatter(tgt_model_tsne[:, 0], x=0, y=1, color=src_labels.astype(str), labels={'color': 'Classes'})


def get_nonexistant_path(fname_path):
    if not os.path.exists(fname_path):
        return fname_path
    filename, file_extension = os.path.splitext(fname_path)
    i = 1
    new_fname = f"{filename}_{i}"
    while os.path.exists(new_fname):
        new_fname = f"{filename}_{i + 1}"
        i+=1
    return new_fname


class L2Norm(nn.Module):
    def __init__(self, n_channels, scale):
        super(L2Norm, self).__init__()
        self.n_channels = n_channels
        self.gamma = scale or None
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(self.n_channels))
        self.reset_parameters()

    def reset_parameters(self):
        init.constant(self.weight, self.gamma)

    def forward(self, x):
        norm = x.pow(2).sum(1).sqrt() + self.eps
        x /= norm.expand_as(x)
        out = self.weight.unsqueeze(0).expand_as(x) * x
        return out


def time_features(data):
    '''计算时域特征参数，输入的data为 样本数*样本长度 的二维数组'''
    f_sum = np.sum(data, axis=-1)  # 总能量
    f_min = np.min(data, axis=-1)  # 最小值
    f_max = np.max(data, axis=-1)  # 最大值
    f_std = np.std(data, axis=-1)  # 标准差
    f_var = np.var(data, axis=-1)  # 方差
    f_pk = f_max - f_min  # 极差
    ent = ant.svd_entropy(data[:,:1].reshape(-1), normalize=True) #熵
    f_avg = np.mean(np.abs(data), axis=-1)  # 整流平均值Mean Absolute
    f_max = np.max(np.abs(data), axis=-1)  # 整流最大值Max Absolute
    # f_sk = np.mean(((data-np.mean(data, axis=-1)[:, :, np.newaxis]) ** 3), axis=-1)/f_std  # 偏度
    s = pd.Series(data)
    f_sk = s.skew() # 偏度 Skewness
    f_ku = np.mean(((data-np.mean(data, axis=-1)[:, :, np.newaxis]) ** 4), axis=-1) / np.power(f_var, 2)  # 峰度Kurtosis
    # math.sqrt(sum([x ** 2 for x in records]) / len(records))
    f_rms = np.norm(data, 2, axis=-1) / np.sqrt(data.shape[-1])  # 均方根
    s = f_rms / f_avg  # 波形因子
    c = f_pk / f_rms  # 峰值因子
    i = f_pk / f_avg  # 脉冲因子
    xr = np.mean(np.sqrt(abs(data)), axis=-1) ** 2
    L = f_pk / xr  # 裕度因子
    return f_sum, f_std, f_avg, ent, f_avg, f_max, f_sk, f_ku, f_rms, s, c, i, L




def reconstruct_u_graph(self, u_seq):
    A = np.zeros((self.opt.num_domain, self.opt.num_domain))

    new_u = to_np(u_seq)
    A = np.zeros((self.num_domain, self.num_domain))
    tmp_size = new_u.shape[1]
    for i in range(self.num_domain):
        for j in range(i + 1, self.num_domain):
            M = ot.dist(new_u[i], new_u[j])
            a = np.ones(tmp_size) / tmp_size
            b = np.ones(tmp_size) / tmp_size
            Wd = ot.emd2(a, b, M)
            A[i][j] = Wd
            A[j][i] = Wd
    bound = np.sort(A.flatten())[int(self.num_domain**2 * 1 / 3)]
    A_dis = A
    # generate self.A
    self.A = (A < bound)

    # calculate the beta seq
    mu_beta = self.embedding.fit_transform(A_dis)
    mu_beta = torch.from_numpy(mu_beta).to(self.device)
    mu_beta = F.normalize(mu_beta)

    return mu_beta


def loss_D_dann(d_seq,domain_seq):
        # this is for DANN
    return F.nll_loss(flat(d_seq),
                        flat(domain_seq))  # , self.u_seq.mean(1)

def loss_D_cida(self, d_seq):
    # this is for CIDA
    # use L1 instead of L2
    return F.l1_loss(flat(d_seq),
                        flat(self.u_seq.detach()))  # , self.u_seq.mean(1)

def loss_D_grda(self, d_seq):
    # this is for GRDA
    A = self.A

    criterion = nn.BCEWithLogitsLoss()
    d = d_seq
    # random pick subchain and optimize the D
    # balance coefficient is calculate by pos/neg ratio
    # A is the adjancency matrix
    sub_graph = sub_graph(my_sample_v=self.opt.sample_v, A=A)

    errorD_connected = torch.zeros((1, )).to(self.device)  # .double()
    errorD_disconnected = torch.zeros((1, )).to(self.device)  # .double()

    count_connected = 0
    count_disconnected = 0

    for i in range(self.opt.sample_v):
        v_i = sub_graph[i]
        # no self loop version!!
        for j in range(i + 1, self.opt.sample_v):
            v_j = sub_graph[j]
            label = torch.full(
                (self.tmp_batch_size, ),
                A[v_i][v_j],
                device=self.device,
            )
            # dot product
            if v_i == v_j:
                idx = torch.randperm(self.tmp_batch_size)
                output = (d[v_i][idx] * d[v_j]).sum(1)
            else:
                output = (d[v_i] * d[v_j]).sum(1)

            if A[v_i][v_j]:  # connected
                errorD_connected += criterion(output, label)
                count_connected += 1
            else:
                errorD_disconnected += criterion(output, label)
                count_disconnected += 1

    # prevent nan
    if count_connected == 0:
        count_connected = 1
    if count_disconnected == 0:
        count_disconnected = 1

    errorD = 0.5 * (errorD_connected / count_connected +
                    errorD_disconnected / count_disconnected)
    # this is a loss balance
    return errorD * num_domain

def sub_graph(self, my_sample_v, A):
    # sub graph tool for grda loss
    if np.random.randint(0, 2) == 0:
        return np.random.choice(num_domain,
                                size=my_sample_v,
                                replace=False)

    # subsample a chain (or multiple chains in graph)
    left_nodes = my_sample_v
    choosen_node = []
    vis = np.zeros(num_domain)
    while left_nodes > 0:
        chain_node, node_num = rand_walk(vis, left_nodes, A)
        choosen_node.extend(chain_node)
        left_nodes -= node_num

    return choosen_node

def rand_walk(self, vis, left_nodes, A):
    # graph random sampling tool for grda loss
    chain_node = []
    node_num = 0
    # choose node
    node_index = np.where(vis == 0)[0]
    st = np.random.choice(node_index)
    vis[st] = 1
    chain_node.append(st)
    left_nodes -= 1
    node_num += 1

    cur_node = st
    while left_nodes > 0:
        nx_node = -1
        node_to_choose = np.where(vis == 0)[0]
        num = node_to_choose.shape[0]
        node_to_choose = np.random.choice(node_to_choose,
                                            num,
                                            replace=False)

        for i in node_to_choose:
            if cur_node != i:
                # have an edge and doesn't visit
                if A[cur_node][i] and not vis[i]:
                    nx_node = i
                    vis[nx_node] = 1
                    chain_node.append(nx_node)
                    left_nodes -= 1
                    node_num += 1
                    break
        if nx_node >= 0:
            cur_node = nx_node
        else:
            break
    return chain_node, node_num

def contrastive_loss(u_con_seq, tmp_batch_size, num_domain, temperature=1):
    u_con_seq = u_con_seq.reshape(tmp_batch_size * num_domain,
                                    -1)
    u_con_seq = nn.functional.normalize(u_con_seq, p=2, dim=1)

    # calculate the cosine similarity between each pair
    logits = torch.matmul(u_con_seq, torch.t(u_con_seq)) / temperature

    # we only choose the one that is:
    # 1, belongs to one domain
    # 2, next to each other
    # as the pair that we want to concentrate them, and all the others will be cancel out

    # the first 2 steps will generate matrix in this format:
    # [0, 1, 0, 0]
    # [0, 0, 1, 0]
    # [0, 0, 0, 1]
    # [1, 0, 0, 0]
    base_m = torch.diag(torch.ones(tmp_batch_size - 1),
                        diagonal=1).to(device)
    base_m[tmp_batch_size - 1, 0] = 1

    # Then we generate the "complementary" matrix in this format:
    # [1, 0, 1, 1]
    # [1, 1, 0, 1]
    # [1, 1, 1, 0]
    # [0, 1, 1, 1]
    # which will be used in the mask
    base_m = torch.ones(tmp_batch_size, tmp_batch_size).to(
        device) - base_m
    # generate the true mask with the base matrix as block.
    # [1, 0, 1, 1, 0, 0, 0, 0 ...]
    # [1, 1, 0, 1, 0, 0, 0, 0 ...]
    # [1, 1, 1, 0, 0, 0, 0, 0 ...]
    # [0, 1, 1, 1, 0, 0, 0, 0 ...]
    # [0, 0, 0, 0, 1, 0, 1, 1 ...]
    # [0, 0, 0, 0, 1, 1, 0, 1 ...]
    # [0, 0, 0, 0, 1, 1, 1, 0 ...]
    # [0, 0, 0, 0, 0, 1, 1, 1 ...]
    masks = torch.block_diag(*([base_m] * num_domain))
    logits = logits - masks * 1e9

    # label: which similarity should maximize. We only maximize the similarity of datapoints that:
    # belongs to one domain
    # next to each other
    label = torch.arange(tmp_batch_size * num_domain).to(
        device)
    label = torch.remainder(label + 1, tmp_batch_size) + label.div(
        tmp_batch_size, rounding_mode='floor') * tmp_batch_size

    loss_u_concentrate = F.cross_entropy(logits, label)
    return loss_u_concentrate


def flat(x):
    n, m = x.shape[:2]
    return x.reshape(n * m, *x.shape[2:])

def set_requires_grad(model, requires_grad=True):
    for param in model.parameters():
        param.requires_grad = requires_grad

def get_dis_loss(dis_fake, dis_real):
    D_loss = torch.mean(dis_fake ** 2) + torch.mean((dis_real - 1) ** 2)
    # D_loss = -torch.mean(torch.log10(1-dis_fake)) - torch.mean(torch.log10(dis_real))
    return D_loss

def to_tensor(x, device="cuda"):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x).to(device)
    else:
        x = x.to(device)
    return x

def u_mean(u_seq):
    re = u_seq.dim() == 3

    if re:
        mu_beta = u_seq.mean(1).detach()
    else:
        mu_beta = u_seq.detach()
    mu_beta_mean = mu_beta.mean(0, keepdim=True)
    mu_beta_std = mu_beta.std(0, keepdim=True)
    mu_beta_std = torch.maximum(mu_beta_std,
                                torch.ones_like(mu_beta_std) * 1e-12)
    mu_beta = (mu_beta - mu_beta_mean) / mu_beta_std
    return mu_beta

def my_cat(new_u_seq):
        # concatenation of local domain index u
        st = new_u_seq[0]
        idx_end = len(new_u_seq)
        for i in range(1, idx_end):
            st = torch.cat((st, new_u_seq[i]), dim=1)
        return st