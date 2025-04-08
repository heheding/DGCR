import numpy as np
import torch
from minepy import cstats
 

def mic(x,y):
    x, y = x.cpu().detach().numpy(), y.cpu().detach().numpy()
    mic_c, tic_c =  cstats(x, y, alpha=9, c=5, est="mic_e")
    mic_c = np.sum(mic_c)
    mic_c = torch.tensor(mic_c).cuda()
    return mic_c
    
    