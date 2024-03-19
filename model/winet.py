import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .modules import InvertibleConv1x1
from .refine import Refine
import torch.nn.init as init
class HinResBlock(nn.Module):
    def __init__(self, in_size, out_size, relu_slope=0.2, use_HIN=True):
        super(HinResBlock, self).__init__()
        self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)

        self.conv_1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_3 = nn.Conv2d(in_size+in_size,out_size,3,1,1)
        if use_HIN:
            self.norm = nn.InstanceNorm2d(out_size // 2, affine=True)
        self.use_HIN = use_HIN

    def forward(self, x):
        resi = self.relu_1(self.conv_1(x))
        out_1, out_2 = torch.chunk(resi, 2, dim=1)
        resi = torch.cat([self.norm(out_1), out_2], dim=1)
        resi = self.relu_2(self.conv_2(resi))
        # input = torch.cat([x,resi],dim=1)
        # out = self.conv_3(input)
        return self.identity(x)+resi

def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


def initialize_weights_xavier(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_normal_(m.weight)
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)
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
    h = torch.zeros([out_batch, out_channel, out_height,
                     out_width]).float().cuda()

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
class WaveOp(nn.Module):
    def __init__(self,channels):
        super(WaveOp,self).__init__()
        self.wt = DWT()
        self.idt = RWT()
        self.ada = nn.Sequential(nn.Conv2d(4*channels,4*channels,3,1,1),nn.ReLU())
        self.extra = nn.Conv2d(channels,channels,3,1,1)
        self.relu = nn.ReLU()
    def forward(self,x):
        x_wf,_ = self.wt(x)
        x_wf = self.ada(x_wf)
        x_wf = self.idt(x_wf)
        x_conv = self.extra(x)
        return self.relu(x_wf+x_conv)+x
class SA(nn.Module):
    def __init__(self):
        super(SA,self).__init__()
        self.conv = nn.Conv2d(2,1,kernel_size=7,padding=3)
    def forward(self,x):
        avca = torch.mean(x,dim=1,keepdim=True)
        maca = torch.max(x,dim=1,keepdim=True)[0]
        com = torch.cat([maca,avca],dim=1)
        sa = self.conv(com)
        return torch.sigmoid(sa)
class FFU(nn.Module):
    def __init__(self):
        super(FFU,self).__init__()
        self.sa = SA()
    def forward(self,dfm,x):
        wei = self.sa(dfm-x)
        out = x+dfm*wei
        return out

class DwtFusionMoudle(nn.Module):
    def __init__(self,channels):
        super(DwtFusionMoudle, self).__init__()
        self.LL_fuse = nn.Sequential(nn.Conv2d(2*channels,channels,1,1,0),nn.LeakyReLU(0.1,inplace=False),
                                      nn.Conv2d(channels,channels,1,1,0))
        self.HL_fuse = nn.Sequential(nn.Conv2d(2*channels,channels,1,1,0),nn.LeakyReLU(0.1,inplace=False),
                                      nn.Conv2d(channels,channels,1,1,0))
        self.LH_fuse = nn.Sequential(nn.Conv2d(2*channels,channels,1,1,0),nn.LeakyReLU(0.1,inplace=False),
                                      nn.Conv2d(channels,channels,1,1,0))
        self.HH_fuse = nn.Sequential(nn.Conv2d(2*channels,channels,1,1,0),nn.LeakyReLU(0.1,inplace=False),
                                      nn.Conv2d(channels,channels,1,1,0))
        self.down  = nn.Conv2d(channels,channels,3,2,1)
        self.downms = nn.Conv2d(channels,channels,3,2,1)
        self.conv1 = nn.Conv2d(channels,channels,3,1,1)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.conv3 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.conv4 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.fu1 = FFU()
        self.fu2 = FFU()
        self.fu3 = FFU()
        self.dwt = DWT()
        self.idwt = IWT()
    def forward(self,msf,panf):
        _,(pan_LL, pan_HL, pan_LH, pan_HH)=self.dwt(panf)
        _,(ms_LL, ms_HL, ms_LH, ms_HH)=self.dwt(msf)

        ll_fused = self.LL_fuse(torch.cat([ms_LL,pan_LL],1))
        hl_fused = self.HL_fuse(torch.cat([ms_HL, pan_HL], 1))
        lh_fused = self.LH_fuse(torch.cat([ms_LH, pan_LH], 1))
        hh_fused = self.HH_fuse(torch.cat([ms_HH, pan_HH], 1))
        fea = self.down(panf)
        for_c = fea-ll_fused
        hl_fused = self.conv1(self.fu1(for_c,hl_fused))
        lh_fused = self.conv2(self.fu2(for_c,lh_fused))
        hh_fused = self.conv3(self.fu3(for_c,hh_fused))
        fused = self.idwt(self.conv4(self.downms(msf)),hl_fused,lh_fused,hh_fused)
        return msf+fused


def downsample(x,h,w):
    pass
def upsample(x, h, w):
    return F.interpolate(x, size=[h, w], mode='bicubic', align_corners=True)
class Net(nn.Module):
    def __init__(self, num_channels=4, channels=None, base_filter=None, args=None):
        super(Net, self).__init__()
        channels = base_filter
        self.conv_ms = nn.Conv2d(num_channels,channels,3,1,1)
        self.convpan = nn.Conv2d(1,channels,3,1,1)
        self.msw = nn.Sequential(HinResBlock(channels,channels),HinResBlock(channels,channels),HinResBlock(channels,channels))
        self.panw = nn.Sequential(HinResBlock(channels,channels),HinResBlock(channels,channels),HinResBlock(channels,channels))
        self.wavef1 = WInvBlock(2*channels, channels)
        self.hfb1 = DwtFusionMoudle(channels)
        self.wavef2 = WInvBlock(2*channels, channels)
        self.hfb2 = DwtFusionMoudle(channels)
        self.wavef3 = WInvBlock(2*channels, channels)
        self.hfb3 = DwtFusionMoudle(channels)


        self.wop1 = HinResBlock(channels,channels)
        self.wop2 = HinResBlock(channels,channels)
        self.wop3 = HinResBlock(channels,channels)
        self.refine = Refine(channels, 4)
    def forward(self,ms,_,pan):


        if type(pan) == torch.Tensor:
            pass
        elif pan == None:
            raise Exception('User does not provide pan image!')
        _, _, m, n = ms.shape
        _, _, M, N = pan.shape
        mHR = upsample(ms, M, N)  # size 4
        ms = mHR
        msf = self.conv_ms(ms)  # (4->channels) # size 4
        panf = self.convpan(pan)  # (1->channels)
        msf  = self.msw(msf)
        panf = self.panw(panf)


        msf = self.wavef1(torch.cat([msf, panf], 1))
        msf = self.hfb1(msf,panf)
        panf = self.wop1(panf)

        msf = self.wavef2(torch.cat([msf, panf], 1))
        msf = self.hfb2(msf,panf)
        panf = self.wop2(panf)

        msf = self.wavef3(torch.cat([msf, panf], 1))
        msf = self.hfb3(msf,panf)

        return self.refine(msf)+ms

class WInvBlock(nn.Module):
    def __init__(self, channel_num, channel_split_num, d = 1, clamp=0.8):
        super(WInvBlock,self).__init__()
        self.channel_num = channel_num
        self.channel_split_num = channel_split_num
        self.fuse = nn.Conv2d(channel_num,channel_split_num,1,1,0)
        self.dwt = DWT()
        self.iwt = WinvIWT()
        self.split_len1 = channel_split_num #LL
        self.split_len2 = 3*channel_split_num #HH
        self.P1 = HinResBlock(self.split_len1, self.split_len2)
        self.U1 = HinResBlock(self.split_len2, self.split_len1)
        self.P2 = HinResBlock(self.split_len1, self.split_len2)
        self.U2 = HinResBlock(self.split_len2, self.split_len1)
        self.flow_permutation = lambda z, logdet, rev: self.invconv(z, logdet, rev)
    def forward(self,x):
        x = self.fuse(x)
        _,(x_LL, x_HL, x_LH, x_HH) = self.dwt(x)
        low = x_LL
        high = torch.cat([x_HL, x_LH, x_HH],dim=1)
        p1 = high-self.P1(low)
        u1 = low+self.U1(p1)
        phres = p1-self.P2(u1) # high fre 3channel
        u_res = self.U2(phres) +u1#low fre 1channel
        LL = u_res
        H = phres
        return self.iwt(LL,H)
import cv2
import os

def feature_save(tensor, name, i=0):
    inp = tensor.cpu().data.numpy().transpose(1, 2, 0)
    # inp = tensor.detach().cpu()
    inp = inp.clip(0,1)
    # inp = inp.squeeze(2)
    if not os.path.exists(name):
        os.makedirs(name)
    for i in range(inp.shape[2]):
        f = inp[:, :, i]*255
        f = cv2.applyColorMap(np.uint8(f), cv2.COLORMAP_SUMMER)
        cv2.imwrite(name + '/' + str(i) + '.png', f)
