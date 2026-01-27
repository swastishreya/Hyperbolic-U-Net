import torch.nn as nn
import torch.utils.checkpoint as cp

from hypll.manifolds.poincare_ball import Curvature, PoincareBall
from hypll.tensors import TangentTensor
from hypll.nn import ChangeManifold

from .unet_parts import (
    DoubleConv,
    Down,
    Up,
    OutConv,
)

class CheckpointModule(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, *inputs):
        def custom_forward(*inputs):
            return self.module(*inputs)
        return cp.checkpoint(custom_forward, *inputs)

class FlexHUNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False, curvature=[0.1, 0.1, 0.1, 0.1, 0.1], trainable=True, init_feats=2, depth=4):
        super(FlexHUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.init_feats = init_feats
        self.depth = depth
        self.feat_dict = {}

        self.manifolds = [PoincareBall(c=Curvature(curv, requires_grad=trainable)) for curv in curvature]
        self.switch_manifolds = [ChangeManifold(target_manifold) for target_manifold in self.manifolds]

        self.inc = DoubleConv(n_channels, init_feats, manifold=self.manifolds[0])
        self.outc = OutConv(init_feats, n_classes, manifold=self.manifolds[0])

        factor = 2 if bilinear else 1
        self.down_blocks = []
        for i in range(depth):
            if i == (depth-1):
                self.down_blocks.append(Down(init_feats*pow(2, i), init_feats*pow(2, i+1) // factor, manifold=self.manifolds[i+1]))
            else:
                self.down_blocks.append(Down(init_feats*pow(2, i), init_feats*pow(2, i+1), manifold=self.manifolds[i+1]))

        self.up_blocks = []
        for i in range(depth, 0, -1):
            if i == 1:
                self.up_blocks.append(Up(init_feats*pow(2, i), init_feats, manifold=self.manifolds[i], bilinear=bilinear))
            else:
                self.up_blocks.append(Up(init_feats*pow(2, i), init_feats*pow(2, i-1) // factor, manifold=self.manifolds[i], bilinear=bilinear))

        self.down_blocks = nn.ModuleList(self.down_blocks)
        self.up_blocks = nn.ModuleList(self.up_blocks)

    def forward(self, x):
        x = TangentTensor(
            data=x,
            manifold_points=None,
            manifold=self.manifolds[0],
            man_dim=1,
        )
        x = self.manifolds[0].expmap(x)
        self.feat_dict[1] = self.inc(x)

        for i, layer in enumerate(self.down_blocks):
            self.feat_dict[i+1] = self.switch_manifolds[i+1](self.feat_dict[i+1])
            self.feat_dict[i+2] = layer(self.feat_dict[i+1])
        for i, layer in enumerate(self.up_blocks):
            if i == 0:
                x = layer(self.feat_dict[self.depth-i+1], self.feat_dict[self.depth-i])
            else:
                x = self.switch_manifolds[self.depth-i](x)
                x = layer(x, self.feat_dict[self.depth-i])
            self.feat_dict[self.depth+i+2] = x

        x = self.switch_manifolds[0](x)
        logits = self.outc(x)
        logits = self.manifolds[0].logmap(x=None, y=logits)
        return logits.tensor

    def use_checkpointing(self):
        """Wrap heavy layers with checkpointing inside nn.Module containers."""
        self.inc = CheckpointModule(self.inc)

        for i, block in enumerate(self.down_blocks):
            self.down_blocks[i] = CheckpointModule(block)

        for i, block in enumerate(self.up_blocks):
            self.up_blocks[i] = CheckpointModule(block)

        self.outc = CheckpointModule(self.outc)

class HUNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False, curvature=0.1, trainable=True, init_feats=2, depth=4):
        super(HUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.init_feats = init_feats
        self.depth = depth
        self.feat_dict = {}

        self.manifold = PoincareBall(c=Curvature(curvature, requires_grad=trainable))

        self.inc = DoubleConv(n_channels, init_feats, manifold=self.manifold)
        self.outc = OutConv(init_feats, n_classes, manifold=self.manifold)

        factor = 2 if bilinear else 1
        self.down_blocks = []
        for i in range(depth):
            if i == (depth-1):
                self.down_blocks.append(Down(init_feats*pow(2, i), init_feats*pow(2, i+1) // factor, manifold=self.manifold))
            else:
                self.down_blocks.append(Down(init_feats*pow(2, i), init_feats*pow(2, i+1), manifold=self.manifold))

        self.up_blocks = []
        for i in range(depth, 0, -1):
            if i == 1:
                self.up_blocks.append(Up(init_feats*pow(2, i), init_feats, manifold=self.manifold, bilinear=bilinear))
            else:
                self.up_blocks.append(Up(init_feats*pow(2, i), init_feats*pow(2, i-1) // factor, manifold=self.manifold, bilinear=bilinear))

        self.down_blocks = nn.ModuleList(self.down_blocks)
        self.up_blocks = nn.ModuleList(self.up_blocks)

    def forward(self, x):
        x = TangentTensor(
            data=x,
            manifold_points=None,
            manifold=self.manifold,
            man_dim=1,
        )
        x = self.manifold.expmap(x)
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
        logits = self.manifold.logmap(x=None, y=logits)
        return logits.tensor
    
    def use_checkpointing(self):
        self.inc = CheckpointModule(self.inc)
        for i, block in enumerate(self.down_blocks):
            self.down_blocks[i] = CheckpointModule(block)
        for i, block in enumerate(self.up_blocks):
            self.up_blocks[i] = CheckpointModule(block)
        self.outc = CheckpointModule(self.outc)
