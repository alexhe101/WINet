"""Wavelet lifting and high-frequency enhancement. Derived from the published WINet implementation at 0101209285d54ea8346715225e04a43b764b138b. Parameter names and active operations are retained."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .wavelets import DWT, RWT, WinvIWT, IWT

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
