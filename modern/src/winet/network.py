"""Checkpoint-compatible WINet. Derived from the published WINet implementation at 0101209285d54ea8346715225e04a43b764b138b. Parameter names and active operations are retained."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import HinResBlock, WInvBlock, DwtFusionMoudle
from .refine import Refine

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

    def forward(self, ms, _, pan):
        ms = F.interpolate(ms, size=pan.shape[-2:], mode="bicubic", align_corners=True)
        msf = self.msw(self.conv_ms(ms))
        panf = self.panw(self.convpan(pan))
        # Existing attribute names remain registered exactly once. wop3 is
        # intentionally retained in state_dict, but never used in forward.
        stages = ((self.wavef1, self.hfb1, self.wop1),
                  (self.wavef2, self.hfb2, self.wop2),
                  (self.wavef3, self.hfb3, None))
        for lift, enhance, pan_update in stages:
            msf = lift(torch.cat([msf, panf], dim=1))
            msf = enhance(msf, panf)
            if pan_update is not None:
                panf = pan_update(panf)
        return self.refine(msf) + ms
