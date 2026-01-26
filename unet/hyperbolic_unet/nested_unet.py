from torch import nn
import torch.utils.checkpoint as cp

from hypll.manifolds.poincare_ball import Curvature, PoincareBall
from hypll.tensors import TangentTensor
import hypll.nn as hnn
from .poincare_ops import poincare_cat
from .upsample import HBilinearUpsample # There's an HUpsample function too, but it is pseudo-hyperbolic

class CheckpointModule(nn.Module):
    """Wrap a module so its forward pass is checkpointed."""
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, *inputs):
        def custom_forward(*inputs):
            return self.module(*inputs)
        return cp.checkpoint(custom_forward, *inputs)

class HVGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, manifold):
        super().__init__()
        self.manifold = manifold
        self.relu = hnn.HReLU(manifold=manifold)
        self.conv1 = hnn.HConvolution2d(in_channels, middle_channels, kernel_size=3, manifold=manifold, padding=1)
        self.bn1 = hnn.HBatchNorm2d(middle_channels, manifold=manifold)
        self.conv2 = hnn.HConvolution2d(middle_channels, out_channels, kernel_size=3, manifold=manifold, padding=1)
        self.bn2 = hnn.HBatchNorm2d(out_channels, manifold=manifold)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out


class HNUNet(nn.Module):
    def __init__(self, n_channels, n_classes, curvature=0.1, trainable=True, init_feats=32, **kwargs):
        super().__init__()

        nb_filter = [init_feats*pow(2, i) for i  in range(5)]
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.curvature = curvature
        self.trainable = trainable
        self.bilinear = True

        self.manifold = PoincareBall(c=Curvature(curvature, requires_grad=trainable))

        self.pool = hnn.HMaxPool2d(2, manifold=self.manifold)
        # self.up = HUpsample(scale_factor=2, mode='bilinear', align_corners=True, manifold=self.manifold)
        self.up = HBilinearUpsample(manifold=self.manifold)

        self.conv0_0 = HVGGBlock(n_channels, nb_filter[0], nb_filter[0], manifold=self.manifold)
        self.conv1_0 = HVGGBlock(nb_filter[0], nb_filter[1], nb_filter[1], manifold=self.manifold)
        self.conv2_0 = HVGGBlock(nb_filter[1], nb_filter[2], nb_filter[2], manifold=self.manifold)
        self.conv3_0 = HVGGBlock(nb_filter[2], nb_filter[3], nb_filter[3], manifold=self.manifold)
        self.conv4_0 = HVGGBlock(nb_filter[3], nb_filter[4], nb_filter[4], manifold=self.manifold)

        self.conv3_1 = HVGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3], manifold=self.manifold)
        self.conv2_2 = HVGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2], manifold=self.manifold)
        self.conv1_3 = HVGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1], manifold=self.manifold)
        self.conv0_4 = HVGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0], manifold=self.manifold)

        self.final = hnn.HConvolution2d(nb_filter[0], n_classes, kernel_size=1, manifold=self.manifold)


    def forward(self, input):
        input = TangentTensor(
            data=input,
            manifold_points=None,
            manifold=self.manifold,
            man_dim=1,
        )
        input = self.manifold.expmap(input)

        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        x3_1 = self.conv3_1(poincare_cat(x3_0, self.up(x4_0), 1))
        x2_2 = self.conv2_2(poincare_cat(x2_0, self.up(x3_1), 1))
        x1_3 = self.conv1_3(poincare_cat(x1_0, self.up(x2_2), 1))
        x0_4 = self.conv0_4(poincare_cat(x0_0, self.up(x1_3), 1))

        output = self.final(x0_4)
        output = self.manifold.logmap(x=None, y=output)
        return output.tensor
    
    def use_checkpointing(self):
        """
        Balanced activation checkpointing for HNUNet.
        Wraps only the expensive HVGGBlock convolution layers.
        """
        # Encoder path
        self.conv0_0 = CheckpointModule(self.conv0_0)
        self.conv1_0 = CheckpointModule(self.conv1_0)
        self.conv2_0 = CheckpointModule(self.conv2_0)
        self.conv3_0 = CheckpointModule(self.conv3_0)
        self.conv4_0 = CheckpointModule(self.conv4_0)
        # Decoder path
        self.conv3_1 = CheckpointModule(self.conv3_1)
        self.conv2_2 = CheckpointModule(self.conv2_2)
        self.conv1_3 = CheckpointModule(self.conv1_3)
        self.conv0_4 = CheckpointModule(self.conv0_4)
        self.final = CheckpointModule(self.final)



class HNestedUNet(nn.Module):
    def __init__(self, n_channels, n_classes, deep_supervision=False, curvature=0.1, trainable=True, init_feats=8, **kwargs):
        super().__init__()

        nb_filter = [init_feats*pow(2, i) for i  in range(5)]
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.curvature = curvature
        self.trainable = trainable
        self.bilinear = True
        self.deep_supervision = deep_supervision

        self.manifold = PoincareBall(c=Curvature(curvature, requires_grad=trainable))

        self.pool = hnn.HMaxPool2d(2, manifold=self.manifold)
        # self.up = HUpsample(scale_factor=2, mode='bilinear', align_corners=True, manifold=self.manifold)
        self.up = HBilinearUpsample(manifold=self.manifold)

        self.conv0_0 = HVGGBlock(n_channels, nb_filter[0], nb_filter[0], manifold=self.manifold)
        self.conv1_0 = HVGGBlock(nb_filter[0], nb_filter[1], nb_filter[1], manifold=self.manifold)
        self.conv2_0 = HVGGBlock(nb_filter[1], nb_filter[2], nb_filter[2], manifold=self.manifold)
        self.conv3_0 = HVGGBlock(nb_filter[2], nb_filter[3], nb_filter[3], manifold=self.manifold)
        self.conv4_0 = HVGGBlock(nb_filter[3], nb_filter[4], nb_filter[4], manifold=self.manifold)

        self.conv0_1 = HVGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0], manifold=self.manifold)
        self.conv1_1 = HVGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1], manifold=self.manifold)
        self.conv2_1 = HVGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2], manifold=self.manifold)
        self.conv3_1 = HVGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3], manifold=self.manifold)

        self.conv0_2 = HVGGBlock(nb_filter[0]*2+nb_filter[1], nb_filter[0], nb_filter[0], manifold=self.manifold)
        self.conv1_2 = HVGGBlock(nb_filter[1]*2+nb_filter[2], nb_filter[1], nb_filter[1], manifold=self.manifold)
        self.conv2_2 = HVGGBlock(nb_filter[2]*2+nb_filter[3], nb_filter[2], nb_filter[2], manifold=self.manifold)

        self.conv0_3 = HVGGBlock(nb_filter[0]*3+nb_filter[1], nb_filter[0], nb_filter[0], manifold=self.manifold)
        self.conv1_3 = HVGGBlock(nb_filter[1]*3+nb_filter[2], nb_filter[1], nb_filter[1], manifold=self.manifold)

        self.conv0_4 = HVGGBlock(nb_filter[0]*4+nb_filter[1], nb_filter[0], nb_filter[0], manifold=self.manifold)

        if self.deep_supervision:
            self.final1 = hnn.HConvolution2d(nb_filter[0], n_classes, kernel_size=1, manifold=self.manifold)
            self.final2 = hnn.HConvolution2d(nb_filter[0], n_classes, kernel_size=1, manifold=self.manifold)
            self.final3 = hnn.HConvolution2d(nb_filter[0], n_classes, kernel_size=1, manifold=self.manifold)
            self.final4 = hnn.HConvolution2d(nb_filter[0], n_classes, kernel_size=1, manifold=self.manifold)
        else:
            self.final = hnn.HConvolution2d(nb_filter[0], n_classes, kernel_size=1, manifold=self.manifold)


    def forward(self, input):
        input = TangentTensor(
            data=input,
            manifold_points=None,
            manifold=self.manifold,
            man_dim=1,
        )
        input = self.manifold.expmap(input)
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(poincare_cat(x0_0, self.up(x1_0), 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(poincare_cat(x1_0, self.up(x2_0), 1))
        x0_2 = self.conv0_2(poincare_cat(x0_0, poincare_cat(x0_1, self.up(x1_1), 1), 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(poincare_cat(x2_0, self.up(x3_0), 1))
        x1_2 = self.conv1_2(poincare_cat(x1_0,  poincare_cat(x1_1, self.up(x2_1), 1), 1))
        x0_3 = self.conv0_3(poincare_cat(x0_0,  poincare_cat(x0_1,  poincare_cat(x0_2, self.up(x1_2), 1), 1), 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(poincare_cat(x3_0, self.up(x4_0), 1))
        x2_2 = self.conv2_2(poincare_cat(x2_0,  poincare_cat(x2_1, self.up(x3_1), 1), 1))
        x1_3 = self.conv1_3(poincare_cat(x1_0,  poincare_cat(x1_1,  poincare_cat(x1_2, self.up(x2_2), 1), 1), 1))
        x0_4 = self.conv0_4(poincare_cat(x0_0,  poincare_cat(x0_1,  poincare_cat(x0_2,  poincare_cat(x0_3, self.up(x1_3), 1), 1), 1), 1))

        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            output1 = self.manifold.logmap(x=None, y=output1)
            output2 = self.manifold.logmap(x=None, y=output2)
            output3 = self.manifold.logmap(x=None, y=output3)
            output4 = self.manifold.logmap(x=None, y=output4)
            return [output1.tensor, output2.tensor, output3.tensor, output4.tensor]

        else:
            output = self.final(x0_4)
            output = self.manifold.logmap(x=None, y=output)
            return output.tensor
        
    def use_checkpointing(self):
        """
        Balanced activation checkpointing for HNestedUNet.
        Wraps the heavy HVGGBlock convolutional layers while skipping
        lightweight manifold ops, pooling, and final output layers.
        """
        # Encoder path
        self.conv0_0 = CheckpointModule(self.conv0_0)
        self.conv1_0 = CheckpointModule(self.conv1_0)
        self.conv2_0 = CheckpointModule(self.conv2_0)
        self.conv3_0 = CheckpointModule(self.conv3_0)
        self.conv4_0 = CheckpointModule(self.conv4_0)

        # Decoder path
        self.conv0_1 = CheckpointModule(self.conv0_1)
        self.conv1_1 = CheckpointModule(self.conv1_1)
        self.conv2_1 = CheckpointModule(self.conv2_1)
        self.conv3_1 = CheckpointModule(self.conv3_1)
        self.conv0_2 = CheckpointModule(self.conv0_2)
        self.conv1_2 = CheckpointModule(self.conv1_2)
        self.conv2_2 = CheckpointModule(self.conv2_2)
        self.conv0_3 = CheckpointModule(self.conv0_3)
        self.conv1_3 = CheckpointModule(self.conv1_3)
        self.conv0_4 = CheckpointModule(self.conv0_4)

        # Optionally wrap finals
        if self.deep_supervision:
            self.final1 = CheckpointModule(self.final1)
            self.final2 = CheckpointModule(self.final2)
            self.final3 = CheckpointModule(self.final3)
            self.final4 = CheckpointModule(self.final4)
        else:
            self.final = CheckpointModule(self.final)
