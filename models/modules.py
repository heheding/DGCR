import torch
import torch.nn as nn
import torch.nn.functional as F
from arg import get_args

args = get_args()
# 800 128
input = args.input
hidden = args.hidden    # 512 800
uz = args.uz # 64 10
uchange = uz
class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class UConcenNet(nn.Module):

    def __init__(self):
        super(UConcenNet, self).__init__()
        nh = 512
        nin = 2
        nout = uz
        self.fc1 = nn.Linear(nin, nh)
        self.fc2 = nn.Linear(nh, nout)

    def forward(self, x):
        re = x.dim() == 3

        if re:
            T, B, C = x.shape
            x = x.reshape(T * B, -1)

        u = F.relu(self.fc1(x.float()))
        u = self.fc2(u)

        if re:
            u = u.reshape(T, B, -1)

        return u
    
class UNva(nn.Module):
    """
    Input: Data X
    Ouput: The estimated domain index
    Using Gaussian model
    """

    def __init__(self):
        super(UNva, self).__init__()
        nh = hidden
        nin = input
        n_u = uz
        nout = n_u
        self.fc1 = nn.Linear(nin, nh)

        self.encoder = nn.Sequential(
            nn.Linear(nh, nh),
            nn.ReLU(inplace=True),
            nn.Linear(nh, nh),
            nn.ReLU(inplace=True),
            nn.Linear(nh, nh),
            nn.ReLU(inplace=True)
        )

        self.fc_mu = nn.Linear(nh, nout)
        self.fc_log_var = nn.Linear(nh, nout)

    def encode(self, x):
        result = self.encoder(x)
        mu = self.fc_mu(result)
        log_var = self.fc_log_var(result)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        u = mu + std * eps
        return u

    def forward(self, x):
        re = x.dim() == 4

        if re:
            T, B, C, D = x.shape
            x = x.reshape(T*B, -1)
        
        if x.dim() == 3:
            T, B, C = x.shape
            x = x.reshape(T, -1)

        # u step is not reshaped!!
        x = F.relu(self.fc1(x))
        mu, log_var = self.encode(x)
        u = self.reparameterize(mu, log_var)

        if re:
            u = u.reshape(T, B, -1)
            mu = mu.reshape(T, B, -1)
            log_var = log_var.reshape(T, B, -1)

        return u, mu, log_var

class UNin(nn.Module):
    """
    Input: Data X
    Ouput: The estimated domain index
    Using Gaussian model
    """

    def __init__(self):
        super(UNin, self).__init__()
        nh = hidden
        nin = input
        n_u = uz
        nout = n_u
        self.fc1 = nn.Linear(nin, nh)

        self.encoder = nn.Sequential(
            nn.Linear(nh, nh),
            nn.ReLU(inplace=True),
            nn.Linear(nh, nh),
            nn.ReLU(inplace=True),
            nn.Linear(nh, nh),
            nn.ReLU(inplace=True)
        )

        self.fc_mu = nn.Linear(nh, nout)
        self.fc_log_var = nn.Linear(nh, nout)

    def encode(self, x):
        result = self.encoder(x)
        mu = self.fc_mu(result)
        log_var = self.fc_log_var(result)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        u = mu + std * eps
        return u

    def forward(self, x):
        re = x.dim() == 4

        if re:
            T, B, C, D = x.shape
            x = x.reshape(T*B, -1)
        
        if x.dim() == 3:
            T, B, C = x.shape
            x = x.reshape(T, -1)

        # u step is not reshaped!!
        x = F.relu(self.fc1(x))
        mu, log_var = self.encode(x)
        u = self.reparameterize(mu, log_var)

        if re:
            u = u.reshape(T, B, -1)
            mu = mu.reshape(T, B, -1)
            log_var = log_var.reshape(T, B, -1)

        return u, mu, log_var

class UNno(nn.Module):
    """
    Input: Data X
    Ouput: The estimated domain index
    Using Gaussian model
    """

    def __init__(self):
        super(UNno, self).__init__()
        nh = hidden
        nin = input
        n_u = uz
        nout = n_u
        self.fc1 = nn.Linear(nin, nh)

        self.encoder = nn.Sequential(
            nn.Linear(nh, nh),
            nn.ReLU(inplace=True),
            nn.Linear(nh, nh),
            nn.ReLU(inplace=True),
            nn.Linear(nh, nh),
            nn.ReLU(inplace=True)
        )

        self.fc_mu = nn.Linear(nh, nout)
        self.fc_log_var = nn.Linear(nh, nout)

    def encode(self, x):
        result = self.encoder(x)
        mu = self.fc_mu(result)
        log_var = self.fc_log_var(result)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        u = mu + std * eps
        return u

    def forward(self, x):
        re = x.dim() == 4

        if re:
            T, B, C, D = x.shape
            x = x.reshape(T*B, -1)
        
        if x.dim() == 3:
            T, B, C = x.shape
            x = x.reshape(T, -1)

        # u step is not reshaped!!
        x = F.relu(self.fc1(x))
        mu, log_var = self.encode(x)
        u = self.reparameterize(mu, log_var)

        if re:
            u = u.reshape(T, B, -1)
            mu = mu.reshape(T, B, -1)
            log_var = log_var.reshape(T, B, -1)

        return u, mu, log_var
        
class UReconstructNet(nn.Module):
    
    def __init__(self):
        super(UReconstructNet, self).__init__()

        nh = hidden
        nu = uchange
        nx = uz  # the dimension of x

        self.fc1 = nn.Linear(nu, int(nh))
        self.fc2 = nn.Linear(int(nh), int(nh))
        self.fc3 = nn.Linear(int(nh), int(nh))
        self.fc_final = nn.Linear(int(nh), nx)

    def forward(self, x):
        re = x.dim() == 3

        if re:
            T, B, C = x.shape
            x = x.reshape(T * B, -1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc_final(x)

        if re:
            x = x.reshape(T, B, -1)
        return x


class ReconstructNet(nn.Module):

    def __init__(self):
        super(ReconstructNet, self).__init__()

        nh = hidden
        nu = uz
        nx = input  # the dimension of x

        self.fc1 = nn.Linear(nu, int(nh))
        self.fc2 = nn.Linear(int(nh)*2, int(nh))
        self.fc3 = nn.Linear(int(nh), int(nh))
        self.fc_final = nn.Linear(int(nh), nx)

    def forward(self, x, y):
        re = x.dim() == 3

        if re:
            T, B, C = x.shape
            x = x.reshape(T * B, -1)
            y = y.reshape(T * B, -1)

        x = F.relu(self.fc1(x))
        y = F.relu(self.fc1(y))
        x = torch.cat((x, y), dim=1)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc_final(x)

        if re:
            x = x.reshape(T, B, 30, -1)
        return x

class ZReconstructNet(nn.Module):

    def __init__(self):
        super(ZReconstructNet, self).__init__()

        nh = hidden
        nu = hidden
        nx = input  # the dimension of x

        self.fc1 = nn.Linear(nu, int(nh))
        self.fc2 = nn.Linear(int(nh), int(nh))
        self.fc3 = nn.Linear(int(nh), int(nh))
        self.fc_final = nn.Linear(int(nh), nx)

    def forward(self, x):
        re = x.dim() == 3

        if re:
            T, B, C = x.shape
            x = x.reshape(T * B, -1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc_final(x)

        if re:
            x = x.reshape(T, B, 30, -1)
        return x


class Q_ZNet_beta(nn.Module):

    def __init__(self):
        super(Q_ZNet_beta, self).__init__()

        nh = hidden
        nu = 2
        nx = input # the dimension of x
        n_beta = 1


        self.fc1 = nn.Linear(nx, nh)
        self.fc2 = nn.Linear(nh * 3, nh * 2)
        self.fc3 = nn.Linear(nh * 2, nh * 2)
        self.fc_final = nn.Linear(nh * 2, nh)

        self.fc1_u = nn.Linear(nu, nh)
        self.fc2_u = nn.Linear(nh, nh)

        self.fc1_beta = nn.Linear(n_beta, nh)
        self.fc2_beta = nn.Linear(nh, nh)

        self.fc_q_mu = nn.Linear(nh, nh)
        self.fc_q_log_var = nn.Linear(nh, nh)

        self.fc_q_mu_2 = nn.Linear(nh, nh)
        self.fc_q_log_var_2 = nn.Linear(nh, nh)

        self.fc_p_mu = nn.Linear(nh, nh)
        self.fc_p_log_var = nn.Linear(nh, nh)

        self.fc_p_mu_2 = nn.Linear(nh, nh)
        self.fc_p_log_var_2 = nn.Linear(nh, nh)

    def encode(self, x, u, beta):
        x = F.relu(self.fc1(x))
        u = F.relu(self.fc1_u(u.float()))
        u = F.relu(self.fc2_u(u))
        # u dim
        # (domain * batch) x h

        beta = F.relu(self.fc1_beta(beta.float()))
        beta = F.relu(self.fc2_beta(beta))
        # # beta dim
        # # dim: domain x h

        # tmp_B = int(u.shape[0] / beta.shape[0])

        # beta = beta.unsqueeze(dim=1).expand(-1, tmp_B,
        #                                     -1).reshape(u.shape[0], -1)
        # # beta dim
        # # (domain * batch) x h

        # combine feature in the middle
        x = torch.cat((x, u, beta), dim=1)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc_final(x))

        q_mu = F.relu(self.fc_q_mu(x))
        q_mu = self.fc_q_mu_2(q_mu)
        q_log_var = F.relu(self.fc_q_log_var(x))
        q_log_var = self.fc_q_log_var_2(q_log_var)

        p_mu = F.relu(self.fc_p_mu(x))
        p_mu = self.fc_p_mu_2(p_mu)
        p_log_var = F.relu(self.fc_p_log_var(x))
        p_log_var = self.fc_p_log_var_2(p_log_var)

        return q_mu, q_log_var, p_mu, p_log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        x = mu + std * eps
        return x

    def forward(self, x, u, beta):
        re = x.dim() == 3

        if re:
            T, B, C = x.shape
            x = x.reshape(T * B, -1)
            u = u.reshape(T * B, -1)
            beta = beta.reshape(T * B, -1)

        q_mu, q_log_var, p_mu, p_log_var = self.encode(x, u, beta)
        q_z = self.reparameterize(q_mu, q_log_var)
        p_z = self.reparameterize(p_mu, p_log_var)

        if re:
            q_z = q_z.reshape(T, B, -1)
            p_z = p_z.reshape(T, B, -1)
            q_mu = q_mu.reshape(T, B, -1)
            q_log_var = q_log_var.reshape(T, B, -1)
            p_mu = p_mu.reshape(T, B, -1)
            p_log_var = p_log_var.reshape(T, B, -1)
        return q_z, q_mu, q_log_var, p_z, p_mu, p_log_var

class SAR(nn.Module):
    """
    Discriminator doing binary classification: source v.s. target
    """

    def __init__(self):
        super(SAR, self).__init__()
        nh = hidden

        nin = uchange
        self.fc3 = nn.Linear(nin, nh)
        self.bn3 = nn.BatchNorm1d(nh)

        self.fc4 = nn.Linear(nh, nh)
        self.bn4 = nn.BatchNorm1d(nh)

        self.fc5 = nn.Linear(nh, nh)
        self.bn5 = nn.BatchNorm1d(nh)

        self.fc6 = nn.Linear(nh, nh)
        self.bn6 = nn.BatchNorm1d(nh)

        self.fc7 = nn.Linear(nh, nh)
        self.bn7 = nn.BatchNorm1d(nh)

        self.fc_final = nn.Linear(nh, 1)

    def forward(self, x):
        re = x.dim() == 3

        if re:
            T, B, C = x.shape
            x = x.reshape(T * B, -1)

        x = F.relu(self.bn3(self.fc3(x)))
        x = F.relu(self.bn4(self.fc4(x)))
        x = F.relu(self.bn5(self.fc5(x)))
        x = F.relu(self.bn6(self.fc6(x)))
        x = F.relu(self.bn7(self.fc7(x)))
        x = self.fc_final(x)

        if re:
            return x.reshape(T, B, -1)
        else:
            return x

class Q_ZNet(nn.Module):
    
    def __init__(self):
        super(Q_ZNet, self).__init__()

        nh = hidden
        nu = uchange
        nx = input  # the dimension of x

        self.fc1 = nn.Linear(nx, nh)
        self.fc2 = nn.Linear(nh * 2, nh * 2)
        self.fc3 = nn.Linear(nh * 2, nh * 2)
        self.fc_final = nn.Linear(nh * 2, nh)

        self.fc1_u = nn.Linear(nu, nh)
        self.fc2_u = nn.Linear(nh, nh)

        self.fc_q_mu = nn.Linear(nh * 2, nh)
        self.fc_q_log_var = nn.Linear(nh * 2, nh)

        self.fc_q_mu_2 = nn.Linear(nh, nh)
        self.fc_q_log_var_2 = nn.Linear(nh, nh)

    def encode(self, x, u):
        x = F.relu(self.fc1(x))
        u = F.relu(self.fc1_u(u.float()))
        u = F.relu(self.fc2_u(u))

        # combine feature in the middle
        x = torch.cat((x, u), dim=1)
        q_mu = F.relu(self.fc_q_mu(x))
        q_mu = self.fc_q_mu_2(q_mu)
        q_log_var = F.relu(self.fc_q_log_var(x))
        q_log_var = self.fc_q_log_var_2(q_log_var)

        return q_mu, q_log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        x = mu + std * eps
        return x

    def forward(self, u, x):
        re = x.dim() == 4

        if re:
            T, B, C ,D= x.shape
            x = x.reshape(T * B, -1)
            u = u.reshape(T * B, -1)

        q_mu, q_log_var = self.encode(x, u)
        q_z = self.reparameterize(q_mu, q_log_var)

        if re:
            q_z = q_z.reshape(T, B, -1)
            q_mu = q_mu.reshape(T, B, -1)
            q_log_var = q_log_var.reshape(T, B, -1)
            
        return q_z, q_mu, q_log_var

class Q_ZNetban1(nn.Module):

    def __init__(self):
        super(Q_ZNetban1, self).__init__()

        nh = hidden
        nu = uchange
        nx = input  # the dimension of x

        self.fc1 = nn.Linear(nx, nh)
        self.fc2 = nn.Linear(nh * 3, nh * 2)
        self.fc3 = nn.Linear(nh * 2, nh * 2)
        self.fc_final = nn.Linear(nh * 2, nh)

        self.fc1_u = nn.Linear(nu, nh)
        self.fc2_u = nn.Linear(nh, nh)
        
        self.fc1_dat = nn.Linear(nu, nh)
        self.fc2_dat = nn.Linear(nh, nh)

        self.fc_q_mu = nn.Linear(nh, nh)
        self.fc_q_log_var = nn.Linear(nh, nh)

        self.fc_q_mu_2 = nn.Linear(nh, nh)
        self.fc_q_log_var_2 = nn.Linear(nh, nh)
        
        self.fc_p_mu = nn.Linear(nh, nh)
        self.fc_p_log_var = nn.Linear(nh, nh)

        self.fc_p_mu_2 = nn.Linear(nh, nh)
        self.fc_p_log_var_2 = nn.Linear(nh, nh)

    def encode(self, x, u, dat):
            
        x = F.relu(self.fc1(x))
        u = F.relu(self.fc1_u(u.float()))
        u = F.relu(self.fc2_u(u))
        
        dat = F.relu(self.fc1_dat(dat.float()))
        dat = F.relu(self.fc2_dat(dat))

        # combine feature in the middle
        x = torch.cat((x, u, dat), dim=1)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc_final(x))

        q_mu = F.relu(self.fc_q_mu(x))
        q_mu = self.fc_q_mu_2(q_mu)
        q_log_var = F.relu(self.fc_q_log_var(x))
        q_log_var = self.fc_q_log_var_2(q_log_var)
        
        p_mu = F.relu(self.fc_p_mu(x))
        p_mu = self.fc_p_mu_2(p_mu)
        p_log_var = F.relu(self.fc_p_log_var(x))
        p_log_var = self.fc_p_log_var_2(p_log_var)

        return q_mu, q_log_var, p_mu, p_log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        x = mu + std * eps
        return x

    def forward(self, x, u, dat):
        re = x.dim() == 4

        if re:
            T, B, C ,D= x.shape
            x = x.reshape(T * B, -1)
            u = u.reshape(T * B, -1)
            dat = dat.reshape(T * B, -1)
        
        if x.dim() == 3:
            T, B, C = x.shape
            x = x.reshape(T, -1)

        q_mu, q_log_var, p_mu, p_log_var = self.encode(x, u, dat)
        q_z = self.reparameterize(q_mu, q_log_var)
        p_z = self.reparameterize(p_mu, p_log_var)

        if re:
            q_z = q_z.reshape(T, B, -1)
            q_mu = q_mu.reshape(T, B, -1)
            q_log_var = q_log_var.reshape(T, B, -1)
            p_z = p_z.reshape(T, B, -1)
            p_mu = p_mu.reshape(T, B, -1)
            p_log_var = p_log_var.reshape(T, B, -1)
            
        return q_z, q_mu, q_log_var, p_z, p_mu, p_log_var

class PredNet1(nn.Module):
    
    def __init__(self):
        # This is for classification task.
        super(PredNet1, self).__init__()
        nh, nc = hidden, 1
        nin = nh
        self.fc3 = nn.Linear(nin, nh)
        self.bn3 = nn.BatchNorm1d(nh)
        self.fc4 = nn.Linear(nh, nh)
        self.bn4 = nn.BatchNorm1d(nh)
        self.fc_final = nn.Linear(nh, nc)

    def forward(self, x):
        re = x.dim() == 3
        if re:
            T, B, C = x.shape
            x = x.reshape(T * B, -1)

        x = F.relu(self.bn3(self.fc3(x)))
        x = F.relu(self.bn4(self.fc4(x)))
        x = self.fc_final(x)

        if re:
            x = x.reshape(T, B, -1)
        x = x.squeeze(-1)
        
        return x

class PredNet2(nn.Module):
    
    def __init__(self):
        # This is for classification task.
        super(PredNet2, self).__init__()
        nh, nc = hidden, 1
        nin = nh
        self.fc3 = nn.Linear(nin, nh)
        self.bn3 = nn.BatchNorm1d(nh)
        self.fc4 = nn.Linear(nh, nh)
        self.bn4 = nn.BatchNorm1d(nh)
        self.fc_final = nn.Linear(nh, nc)

    def forward(self, x):
        re = x.dim() == 3
        if re:
            T, B, C = x.shape
            x = x.reshape(T * B, -1)

        x = F.relu(self.bn3(self.fc3(x)))
        x = F.relu(self.bn4(self.fc4(x)))
        x = self.fc_final(x)

        if re:
            x = x.reshape(T, B, -1)
        x = x.squeeze(-1)
        
        return x

class PredNet3(nn.Module):
    
    def __init__(self):
        # This is for classification task.
        super(PredNet3, self).__init__()
        nh, nc = hidden, 1
        nin = nh
        self.fc3 = nn.Linear(nin, nh)
        self.bn3 = nn.BatchNorm1d(nh)
        self.fc4 = nn.Linear(nh, nh)
        self.bn4 = nn.BatchNorm1d(nh)
        self.fc_final = nn.Linear(nh, nc)

    def forward(self, x):
        re = x.dim() == 3
        if re:
            T, B, C = x.shape
            x = x.reshape(T * B, -1)

        x = F.relu(self.bn3(self.fc3(x)))
        x = F.relu(self.bn4(self.fc4(x)))
        x = self.fc_final(x)

        if re:
            x = x.reshape(T, B, -1)
        x = x.squeeze(-1)
        
        return x

class PredNet4(nn.Module):
    
    def __init__(self):
        # This is for classification task.
        super(PredNet4, self).__init__()
        nh, nc = hidden, 1
        nin = nh
        self.fc3 = nn.Linear(nin, nh)
        self.bn3 = nn.BatchNorm1d(nh)
        self.fc4 = nn.Linear(nh, nh)
        self.bn4 = nn.BatchNorm1d(nh)
        self.fc_final = nn.Linear(nh, nc)

    def forward(self, x):
        re = x.dim() == 3
        if re:
            T, B, C = x.shape
            all_feature = x.reshape(T * B, -1)

        x = F.relu(self.bn3(self.fc3(x)))
        x = F.relu(self.bn4(self.fc4(x)))
        x = self.fc_final(x)

        if re:
            x = x.reshape(T, B, -1)
        x = x.squeeze(-1)
        
        return x

class PredNet5(nn.Module):
    
    def __init__(self):
        # This is for classification task.
        super(PredNet5, self).__init__()
        nh, nc = hidden, 1
        nin = nh
        self.fc3 = nn.Linear(nin, nh)
        self.bn3 = nn.BatchNorm1d(nh)
        self.fc4 = nn.Linear(nh, nh)
        self.bn4 = nn.BatchNorm1d(nh)
        self.fc_final = nn.Linear(nh, nc)

    def forward(self, x):
        re = x.dim() == 3
        if re:
            T, B, C = x.shape
            x = x.reshape(T * B, -1)

        x = F.relu(self.bn3(self.fc3(x)))
        x = F.relu(self.bn4(self.fc4(x)))
        x = self.fc_final(x)

        if re:
            x = x.reshape(T, B, -1)
        x = x.squeeze(-1)
        
        return x

class PredNet6(nn.Module):
    
    def __init__(self):
        # This is for classification task.
        super(PredNet6, self).__init__()
        nh, nc = hidden, 1
        nin = nh
        self.fc3 = nn.Linear(nin, nh)
        self.bn3 = nn.BatchNorm1d(nh)
        self.fc4 = nn.Linear(nh, nh)
        self.bn4 = nn.BatchNorm1d(nh)
        self.fc_final = nn.Linear(nh, nc)

    def forward(self, x):
        re = x.dim() == 3
        if re:
            T, B, C = x.shape
            all_feature = x.reshape(T * B, -1)

        x = F.relu(self.bn3(self.fc3(x)))
        x = F.relu(self.bn4(self.fc4(x)))
        x = self.fc_final(x)

        if re:
            x = x.reshape(T, B, -1)
        x = x.squeeze(-1)
        
        return x


