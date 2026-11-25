"""Released Haar transforms and band ordering. Derived from the published WINet implementation at 0101209285d54ea8346715225e04a43b764b138b. Parameter names and active operations are retained."""
import torch
import torch.nn as nn
import torch.nn.functional as F

def dwt_init(x):

    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4
    return torch.cat((x_LL, x_HL, x_LH, x_HH), 1),(x_LL,x_HL,x_LH,x_HH)
def iwt_init(x):
    r = 2
    in_batch, in_channel, in_height, in_width = x.size()
    #print([in_batch, in_channel, in_height, in_width])
    out_batch, out_channel, out_height, out_width = in_batch,int(in_channel/4),r * in_height, r * in_width
    x1 = x[:,0:out_channel, :, :] / 2
    x2 = x[:,out_channel:out_channel * 2, :, :] / 2
    x3 = x[:,out_channel * 2:out_channel * 3, :, :] / 2
    x4 = x[:,out_channel * 3:out_channel * 4, :, :] / 2
    h = torch.zeros((out_batch, out_channel, out_height, out_width),
                    device=x.device, dtype=torch.float32)

    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4

    return h


def iwt_init_g(x_LL,x_HL,x_LH,x_HH):
    x = torch.cat((x_LL, x_HL, x_LH, x_HH), 1)
    return iwt_init(x)
class DWT(nn.Module):
    def __init__(self):
        super(DWT,self).__init__()
        self.requires_grad = False
    def forward(self,x):
        return dwt_init(x)
class RWT(nn.Module):
    def __init__(self):
        super(RWT, self).__init__()
        self.requires_grad = False
    def forward(self,X):
        return iwt_init(X)
class WinvIWT(nn.Module):
    def __init__(self):
        super(WinvIWT, self).__init__()
        self.requires_grad = False
    def forward(self,LL,HH):
        return iwt_init(torch.cat([LL,HH],dim=1))
class IWT(nn.Module):
    def __init__(self):
        super(IWT,self).__init__()
        self.requires_grad = False
    def forward(self,x_LL,x_HL,x_LH,x_HH):
        x = torch.cat((x_LL, x_HL, x_LH, x_HH), 1)
        return iwt_init(x)
