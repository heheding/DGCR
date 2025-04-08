import torch.nn as nn
import torch
from models.functions import ReverseLayerF
import torch.nn.functional as F
# from models.loss import Cmd
from einops import rearrange, repeat

# def entropy(predictions: torch.Tensor, reduction='mean') -> torch.Tensor:
#     r"""Entropy of prediction.
#     The definition is:

#     .. math::
#         entropy(p) = - \sum_{c=1}^C p_c \log p_c

#     where C is number of classes.

#     Args:
#         predictions (tensor): Classifier predictions. Expected to contain raw, normalized scores for each class
#         reduction (str, optional): Specifies the reduction to apply to the output:
#           ``'none'`` | ``'mean'``. ``'none'``: no reduction will be applied,
#           ``'mean'``: the sum of the output will be divided by the number of
#           elements in the output. Default: ``'mean'``

#     Shape:
#         - predictions: :math:`(minibatch, C)` where C means the number of classes.
#         - Output: :math:`(minibatch, )` by default. If :attr:`reduction` is ``'mean'``, then scalar.
#     """
#     epsilon = 1e-5
#     H = -predictions * torch.log(predictions + epsilon)
#     H = H.sum(dim=1)
#     if reduction == 'mean':
#         return H.mean()
#     else:
#         return H
class FeatureNet(nn.Module):
    def __init__(self):
        super(FeatureNet, self).__init__()
        nx, nh = 474, 800
        self.fc1 = nn.Linear(nx, nh)
        self.fc2 = nn.Linear(nh * 2, nh * 2)
        self.fc3 = nn.Linear(nh * 2, nh * 2)
        self.fc4 = nn.Linear(nh * 2, nh * 2)
        self.fc_final = nn.Linear(nh * 2, nh)

        self.fc1_var = nn.Linear(1, nh)
        self.fc2_var = nn.Linear(nh, nh)

    def forward(self, x, t):
        re = x.dim() == 3
        if re:
            T, B, C = x.shape
            x = x.reshape(T * B, -1)
            t = t.reshape(T * B, -1)

        x = F.relu(self.fc1(x))
        t = F.relu(self.fc1_var(t))
        t = F.relu(self.fc2_var(t))

        # combine feature in the middle
        x = torch.cat((x, t), dim=1)

        # main
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc_final(x)

        if re:
            return x.reshape(T, B, -1)
        else:
            return x

class PredNet(nn.Module):
    def __init__(self):
        super(PredNet, self).__init__()

        nh, nc = 800, 1
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
    
class Predcnn(nn.Module):
    def __init__(self):
        super(Predcnn, self).__init__()

        nh, nc = 800, 1
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

        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc_final(x)

        if re:
            x = x.reshape(T, B, -1)
        x = x.squeeze(-1)
        return x

class DiscNet(nn.Module):
    """
    Discriminator doing binary classification: source v.s. target
    """

    def __init__(self):
        super(DiscNet, self).__init__()
        nh = 800

        nin = 800
        self.fc3 = nn.Linear(nin, nh)
        self.bn3 = nn.BatchNorm1d(nh)

        self.fc4 = nn.Linear(nh, nh)
        self.bn4 = nn.BatchNorm1d(nh)

        # self.fc5 = nn.Linear(nh, nh)
        # self.bn5 = nn.BatchNorm1d(nh)

        # self.fc6 = nn.Linear(nh, nh)
        # self.bn6 = nn.BatchNorm1d(nh)

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
        # x = F.relu(self.bn5(self.fc5(x)))
        # x = F.relu(self.bn6(self.fc6(x)))
        x = F.relu(self.bn7(self.fc7(x)))
        x = torch.sigmoid((self.fc_final(x)))

        if re:
            return x.reshape(T, B, -1)
        else:
            return x

class DCNN(nn.Module):
    """
    Discriminator doing binary classification: source v.s. target
    """

    def __init__(self):
        super(DCNN, self).__init__()
        nh = 512

        nin = 800
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
        # x = F.relu(self.bn5(self.fc5(x)))
        # x = F.relu(self.bn6(self.fc6(x)))
        # x = F.relu(self.bn7(self.fc7(x)))
        x = torch.sigmoid((self.fc_final(x)))

        if re:
            return x.reshape(T, B, -1)
        else:
            return x

###########################################################################
class SPF(nn.Module):
    def __init__(self):
        super(SPF, self).__init__()
        nx, nh = 474, 800
        self.fc1 = nn.Linear(nx, nh)
        self.fc2 = nn.Linear(nh * 2, nh * 2)
        self.fc3 = nn.Linear(nh * 2, nh * 2)
        self.fc4 = nn.Linear(nh * 2, nh * 2)
        self.fc_final = nn.Linear(nh * 2, nh)

        self.fc1_var = nn.Linear(1, nh)
        self.fc2_var = nn.Linear(nh, nh)

    def forward(self, x, t):
        re = x.dim() == 3
        if re:
            T, B, C = x.shape
            x = x.reshape(T * B, -1)
            t = t.reshape(T * B, -1)

        x = F.relu(self.fc1(x))
        t = F.relu(self.fc1_var(t))
        t = F.relu(self.fc2_var(t))

        # combine feature in the middle
        x = torch.cat((x, t), dim=1)

        # main
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc_final(x)

        if re:
            return x.reshape(T, B, -1)
        else:
            return x

class SAR(nn.Module):
    """
    Discriminator doing binary classification: source v.s. target
    """

    def __init__(self):
        super(SAR, self).__init__()
        nh = 800

        nin = 800
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

####################################################################
class Feature(nn.Module):
    def __init__(self):
        super(Feature, self).__init__()
        self.gru1 = nn.GRU(1, 32, 2, batch_first=True)
        self.fc_1 = nn.Linear(32, 128)

    def forward(self, x):
        feat, _ = self.gru1(x)
        x = F.relu(self.fc_1(feat[:, -1, :]))
        # x = x.view(x.size(0), 48*4*4)
        return x

class Predictor(nn.Module):
    def __init__(self, prob=0.5):
        super(Predictor, self).__init__()
        self.fc1 = nn.Linear(128, 32)
        self.bn1_fc = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(32, 32) #64 64
        self.bn2_fc = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(32, 1) #64 1
        self.prob = prob

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = x.squeeze(-1)
        return x




class GradReverse(torch.autograd.Function):
    """
    Extension of grad reverse layer
    """
    @staticmethod
    def forward(ctx, x, constant):
        ctx.constant = constant
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.neg() * ctx.constant
        return grad_output, None

    def grad_reverse(x, constant):
        return GradReverse.apply(x, constant)

class Domain_classifier(nn.Module):

    def __init__(self):
        super(Domain_classifier, self).__init__()
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.bce = lambda input, target, weight: \
            F.binary_cross_entropy(input, target, weight=weight, reduction='mean')

    def forward(self, f_s, f_t, constant):
        input = torch.cat((f_s, f_t), dim=0)
        input = GradReverse.grad_reverse(input, constant)
        logits = F.relu(self.fc1(input))
        logits = F.relu(self.fc2(logits))
        logits = torch.sigmoid(self.fc3(logits)).double()

        label_src = torch.ones((f_s.size(0),1)).cuda()
        label_tgt = torch.zeros((f_t.size(0),1)).cuda()
        label_concat = torch.cat((label_src, label_tgt), 0).double()

        w = torch.ones_like(label_concat).double()

        loss = self.bce(logits, label_concat, w.view_as(logits))

        return loss

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        ninx = 1
        nh = 32
        
        self.encoder = nn.Sequential(
            nn.Conv1d(ninx, nh, kernel_size=3, stride=1, padding=1, dilation=1), #128*32*30
            nn.BatchNorm1d(nh),
            nn.ReLU(),
            nn.Conv1d(nh, nh, kernel_size=3, stride=1, padding=1, dilation=1), #128*32*30
            nn.BatchNorm1d(nh),
            nn.ReLU(),
            nn.Conv1d(nh, nh, kernel_size=3, stride=1, padding=1, dilation=1), #128*32*30
            nn.BatchNorm1d(nh)
            ) 
        self.med_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(nh*468, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU())
       

    def forward(self, x):
        

        x = x.permute(0, 2, 1)
    
        x = self.encoder(x)
        x = x.view(-1, 32*204)
        x = self.med_layer(x)

        return x

    
class CNN_SL_bn(nn.Module):
    def __init__(self, num=1):
        super(CNN_SL_bn, self).__init__()
        self.num = num
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1, dilation=1), #128*32*30
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=3, stride=1, padding=1, dilation=1), #128*32*30
            nn.BatchNorm1d(32),
            nn.ReLU(),
            # nn.Conv1d(32, 32, kernel_size=3, stride=1, padding=1, dilation=1), #128*32*30
            # nn.BatchNorm1d(32),
            # nn.ReLU(),
            # nn.Conv1d(32, 32, kernel_size=3, stride=1, padding=1, dilation=1), #128*32*30
            # nn.BatchNorm1d(32),
            # nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=3, stride=1, padding=1, dilation=1), #128*32*30
            nn.BatchNorm1d(32)
            ) 
        self.med_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32*34, 128),
            nn.ReLU())
        self.reshape = nn.Sequential(
            nn.Linear(34*self.num, 34*self.num),
            nn.ReLU(),
            nn.Linear(34*self.num, 34),
            nn.ReLU()
            )
        # self.Classifier = nn.Sequential(
        #     nn.Linear(1024, 1024),
        #     nn.ReLU(),
        #     nn.Linear(1024, 128))        

    def forward(self, src):
        # reshape input (batch_size, input_dim, sequence length)
        # src = src.view(src.size(0), self.input_dim, -1)
        src = src.permute(0, 2, 1)
        if self.num != 1:
            src = src.view(-1,34*self.num)
            src = self.reshape(src)
            src = src.view(-1,1,34)
        # src = src.type(torch.FloatTensor)
        full_features = self.encoder(src)
        full_features = full_features.view(-1, 32*34)
        features = self.med_layer(full_features)
        return features

class SDACNN(nn.Module):
    def __init__(self):
        super(SDACNN, self).__init__()
        ninx = 468
        nint = 1
        nh = 64
        nout = 800

        self.cnnx = nn.Sequential(
            nn.Conv1d(ninx, nh, kernel_size=3, stride=1, padding=1, dilation=1), #128*32*30
            nn.BatchNorm1d(nh),
            nn.ReLU()
            ) 
        self.cnnt = nn.Sequential(
            nn.Conv1d(nint, nh, kernel_size=3, stride=1, padding=1, dilation=1), #128*32*30
            nn.BatchNorm1d(nh),
            nn.ReLU(),
            nn.Conv1d(nh, nh, kernel_size=3, stride=1, padding=1, dilation=1), #128*32*30
            nn.BatchNorm1d(nh),
            nn.ReLU()
            ) 
        self.encoder = nn.Sequential(
            nn.Conv1d(2*nh, 2*nh, kernel_size=3, stride=1, padding=1, dilation=1), #128*32*30
            nn.BatchNorm1d(2*nh),
            nn.ReLU(),
            nn.Conv1d(2*nh, 2*nh, kernel_size=3, stride=1, padding=1, dilation=1), #128*32*30
            nn.BatchNorm1d(2*nh),
            nn.ReLU(),
            nn.Conv1d(2*nh, 2*nh, kernel_size=3, stride=1, padding=1, dilation=1), #128*32*30
            nn.BatchNorm1d(2*nh),
            nn.ReLU(),
            nn.Conv1d(2*nh, 2*nh, kernel_size=3, stride=1, padding=1, dilation=1), #128*32*30
            nn.BatchNorm1d(2*nh)
            ) 
        self.med_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, nout),
            nn.ReLU()
            )     

    def forward(self, x, t):
        # reshape input (batch_size, input_dim, sequence length)
        re = x.dim() == 3
        if re:
            T, B, C = x.shape
            x = x.permute(0, 2, 1)
            t = t.permute(0, 2, 1)
        else:
            T, B = x.shape
            x = x.reshape(1, T, B)
            t = t.reshape(1, T, 1)
            x = x.permute(0, 2, 1)
            t = t.permute(0, 2, 1)
        x = self.cnnx(x)
        t = self.cnnt(t)
        x = torch.cat((x, t), dim=1)
        # src = src.type(torch.FloatTensor)
        full_features = self.encoder(x)
        full_features = full_features.view(-1, 128)
        features = self.med_layer(full_features)
        if re:
            features = features.reshape(T, B, -1)
        else:
            features = features.reshape(T, -1)
        return features

class CNNLSTM(nn.Module):
    def __init__(self):
        super(CNNLSTM, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(7, 32, kernel_size=5, stride=1, padding=1, dilation=1), #128*32*50 tfd3-4 11 128
            nn.BatchNorm1d(32),
            nn.ReLU(),
            # nn.Conv1d(32, 32, kernel_size=3, stride=1, padding=1, dilation=1),# tfd3-4 3 32
            # nn.BatchNorm1d(32),
            # nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=5, stride=1, padding=1, dilation=1),# tfd3-4 3 32
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=3, stride=1, padding=1, dilation=1),# tfd3-4 3 32
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=9, stride=1, padding=1, dilation=1), #128*32*45 tfd3-4 11
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.LSTM(20, 32, 2, batch_first=True, bidirectional=True))
        self.med_layer = nn.Sequential(
            nn.Flatten(),
            # nn.Linear(32*22, 1024),
            # nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU())

    def forward(self, src):
        # reshape input (batch_size, input_dim, sequence length)
        # src = src.view(src.size(0), self.input_dim, -1)
        src = src.permute(0, 2, 1)
        # src = src.type(torch.FloatTensor)
        full_features,_ = self.encoder(src)
        full_features = full_features[:, -1, :]
        features = self.med_layer(full_features)
        return features

class BiLSTM(nn.Module):
    def __init__(self, input_size=8, hidden_size=32, num_layers=3, output_size=128):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.num_directions = 1
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True, bidirectional=False)
        self.linear = nn.Linear(self.num_directions * self.hidden_size, self.output_size)

    def forward(self, input_seq):
        output, _ = self.lstm(input_seq)
        output = output[:, -1, :]
        pred = F.relu(self.linear(output))  # pred()
        return pred


class Att(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout),
            nn.Linear(dim, 1)
        )

    def forward(self, x, mask=None):
        x = torch.unsqueeze(x, 0)
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, float('-inf'))
            del mask

        attn = dots.softmax(dim=-1)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        out = torch.squeeze(out, 0)
        return out

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)



class Attention(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, float('-inf'))
            del mask

        attn = dots.softmax(dim=-1)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads=heads, dropout=dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)))
            ]))

    def forward(self, x, mask=None):
        for attn, ff in self.layers:
            x = attn(x, mask=mask)
            x = ff(x)
        return x


class Seq_Transformer(nn.Module):
    def __init__(self, *, patch_size, dim, depth, heads, mlp_dim, channels=1, dropout=0., emb_dropout=0.):
        super().__init__()
        # num_patches = (seq_len // patch_size)  # ** 2
        patch_dim = channels * patch_size  # ** 2
        # assert num_patches > MIN_NUM_PATCHES, f'your number of patches ({num_patches}) is way too small for attention to be effective. try decreasing your patch size'

        # self.patch_size = patch_size

        # self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        # self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, mlp_dim, dropout)

        self.to_cls_token = nn.Identity()



    def forward(self, forward_seq):
        x = self.patch_to_embedding(forward_seq)
        # print(x.shape)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        # x += self.pos_embedding[:, :(n + 1)]
        # x = self.dropout(x)
        x = self.transformer(x)
        c_t = self.to_cls_token(x[:, 0])

        return c_t


class Discriminator_AR(nn.Module):
    """Discriminator model for source domain."""
    def __init__(self):
        """Init discriminator."""
        super(Discriminator_AR, self).__init__()

        self.AR_disc = nn.GRU(128, 64, 2, batch_first=True)
        self.DC = nn.Linear(64, 1)
    def forward(self, input):
        """Forward the discriminator."""
        # src_shape = [batch_size, seq_len, input_dim]
        input = input.view(input.size(0),-1, 128 )
        encoder_outputs, (encoder_hidden) = self.AR_disc(input)
        features = F.relu(encoder_outputs[:, -1, :])
        domain_output = self.DC(features)
        return domain_output
    def get_parameters(self):
        parameter_list = [{"params":self.AR_disc.parameters(), "lr_mult":0.01, 'decay_mult':1}, {"params":self.DC.parameters(), "lr_mult":0.01, 'decay_mult':1},]
        return parameter_list

class Discriminator_ATT(nn.Module):
    """Discriminator model for source domain."""
    def __init__(self):
        """Init discriminator."""
        super(Discriminator_ATT, self).__init__()
        self.transformer= Seq_Transformer(patch_size=128, dim=64, depth=8, heads= 2 , mlp_dim=256)
        self.DC = nn.Linear(64, 1)
        self.sig = nn.Sigmoid()
    def forward(self, input):
        """Forward the discriminator."""
        # src_shape = [batch_size, seq_len, input_dim]
        input = input.view(input.size(0),-1, 128 )
        features = self.transformer(input)
        domain_output = self.DC(features)
        return domain_output