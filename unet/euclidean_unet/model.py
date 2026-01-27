import torch.nn as nn
import torch.utils.checkpoint as cp

from .unet_parts import (
    DoubleConv,
    Down,
    Up,
    OutConv,
)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False, init_feats=2, depth=4):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.init_feats = init_feats
        self.depth = depth
        self.feat_dict = {}

        self.inc = (DoubleConv(n_channels, init_feats))
        self.outc = (OutConv(init_feats, n_classes))

        factor = 2 if bilinear else 1
        self.down_blocks = []
        for i in range(depth):
            if i == (depth-1):
                self.down_blocks.append(Down(init_feats*pow(2, i), init_feats*pow(2, i+1) // factor))
            else:
                self.down_blocks.append(Down(init_feats*pow(2, i), init_feats*pow(2, i+1)))

        self.up_blocks = []
        for i in range(depth, 0, -1):
            if i == 1:
                self.up_blocks.append(Up(init_feats*pow(2, i), init_feats, bilinear))
            else:
                self.up_blocks.append(Up(init_feats*pow(2, i), init_feats*pow(2, i-1) // factor, bilinear))

        self.down_blocks = nn.ModuleList(self.down_blocks)
        self.up_blocks = nn.ModuleList(self.up_blocks)

    def forward(self, x):
        self.feat_dict[1] = self.inc(x)

        for i, layer in enumerate(self.down_blocks):
            self.feat_dict[i+2] = layer(self.feat_dict[i+1])
        for i, layer in enumerate(self.up_blocks):
            if i == 0:
                x = layer(self.feat_dict[self.depth-i+1], self.feat_dict[self.depth-i])
                self.feat_dict[self.depth+i+1] = x
            else:
                x = layer(x, self.feat_dict[self.depth-i])
                self.feat_dict[self.depth+i+1] = x

        logits = self.outc(x)
        return logits
    
    # def use_checkpointing(self):
    #     """
    #     Enable activation checkpointing for UNet's down and up blocks
    #     to reduce memory usage at the cost of extra compute during backward.
    #     """
    #     def checkpoint_module(module):
    #         def custom_forward(*inputs):
    #             return module(*inputs)
    #         return lambda *inputs: cp.checkpoint(custom_forward, *inputs)

    #     for i, block in enumerate(self.down_blocks):
    #         self.down_blocks[i] = checkpoint_module(block)

    #     for i, block in enumerate(self.up_blocks):
    #         self.up_blocks[i] = checkpoint_module(block)