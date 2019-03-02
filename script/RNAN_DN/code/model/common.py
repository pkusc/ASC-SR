import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)

class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean, rgb_std, sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        self.requires_grad = False

class BasicBlock(nn.Sequential):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, bias=False,
        bn=True, act=nn.ReLU(True)):

        m = [nn.Conv2d(
            in_channels, out_channels, kernel_size,
            padding=(kernel_size//2), stride=stride, bias=bias)
        ]
        if bn: m.append(nn.BatchNorm2d(out_channels))
        if act is not None: m.append(act)
        super(BasicBlock, self).__init__(*m)

class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: m.append(nn.BatchNorm2d(n_feat))
            if i == 0: m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res

class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feat, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feat, 4 * n_feat, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn: m.append(nn.BatchNorm2d(n_feat))
                if act: m.append(act())
        elif scale == 3:
            m.append(conv(n_feat, 9 * n_feat, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn: m.append(nn.BatchNorm2d(n_feat))
            if act: m.append(act())
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)

## add CALayer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        x = self.avg_pool(x)
        x = self.conv_du(x)
        return x

## add CABLayer
class CABLayer(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size, reduction=16,
        bias=True, bn=False, act=nn.ReLU(True)):
        super(CABLayer, self).__init__()
        # CA branch
        modules_ca = []
        modules_ca.append(CALayer(n_feats, reduction))
        # residual branch
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feats))
            if i == 0: modules_body.append(act)

        self.ca = nn.Sequential(*modules_ca)
        self.body = nn.Sequential(*modules_body)


    def forward(self, x):
        x_ca = self.ca(x)
        x_res = self.body(x)
        x_res_ca = x_ca * x_res
        output = x + x_res_ca

        return output

## add CADownLayer
class CADownLayer(nn.Module):
    def __init__(
        self, n_feats, kernel_size, reduction=16):
        super(CADownLayer, self).__init__()
        # CA branch
        modules_ca = []
        modules_ca.append(CALayer(n_feats, reduction))
        # downsample branch
        modules_down = []
        modules_down.append(nn.Conv2d(n_feats,n_feats, 3, stride=2, padding=1))

        self.ca = nn.Sequential(*modules_ca)
        self.down = nn.Sequential(*modules_down)

    def forward(self, x):
        x_ca = self.ca(x)
        x_down = self.down(x)
        x_down_ca = x_ca * x_down

        return x_down_ca

## add CAUpLayer
class CAUpLayer(nn.Module):
    def __init__(
        self, n_feats, kernel_size, reduction=16):
        super(CAUpLayer, self).__init__()
        # CA branch
        modules_ca = []
        modules_ca.append(CALayer(n_feats, reduction))
        # upsample branch
        modules_up = []
        modules_up.append(nn.ConvTranspose2d(n_feats,n_feats, 6, stride=2, padding=2))

        self.ca = nn.Sequential(*modules_ca)
        self.up = nn.Sequential(*modules_up)

    def forward(self, x):
        x_ca = self.ca(x)
        x_up = self.up(x)
        x_up_ca = x_ca * x_up

        return x_up_ca

# add NonLocalBlock2D
# reference: https://github.com/AlexHex7/Non-local_pytorch/blob/master/lib/non_local_simple_version.py
class NonLocalBlock2D(nn.Module):
    def __init__(self, in_channels, inter_channels):
        super(NonLocalBlock2D, self).__init__()
        print('in_channels=' + str(in_channels) + ', inter_channels=' + str(inter_channels))
        # determine channels
        self.in_channels = in_channels
        self.inter_channels = inter_channels
        # add g function
        self.g = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0)
        # add W, used to upscale output channels
        self.W = nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1, stride=1, padding=0)
        # initialize W as zero, so that the initial state of the nonlocal block is an identity mapping
        nn.init.constant(self.W.weight, 0)
        nn.init.constant(self.W.bias, 0)
        # add theta function
        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0)
        # add phi function
        self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        '''
        :param x: (b, 2c, h, w)
        :return:
        '''
        batch_size = x.size(0)
        # (b, 2c, h, w) --> (b, c, h, w) --> (b, c, hw)
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        # (b, c, hw) --> (b, hw, c)
        g_x = g_x.permute(0,2,1)
        # (b, 2c, h, w) --> (b, c, h, w) --> (b, c, hw)
        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        # (b, c, hw)  --> (b, hw, c)
        theta_x = theta_x.permute(0,2,1)
        # (b, 2c, h, w) --> (b, c, h, w) --> (b, c, hw)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        # (b, hw, c) x (b, c, hw) = (b, hw, hw)
        f = torch.matmul(theta_x, phi_x)
        # (b, hw, hw) normalization
        f_div_C = F.softmax(f, dim=1)
        
        # (b, hw, hw) x (b, hw, c) = (b, hw, c)
        y = torch.matmul(f_div_C, g_x)
        # (b, hw, c) --> (b, c, hw)
        y = y.permute(0,2,1).contiguous()
        # 
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z

## define SELayer
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

## define SEResBlock
class SEResBlock(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size, reduction,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(SEResBlock, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(SELayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res

## define trunk branch
class TrunkBranch(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(TrunkBranch, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(ResBlock(conv, n_feat, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1))
        self.body = nn.Sequential(*modules_body)
    
    def forward(self, x):
        tx = self.body(x)

        return tx

## define mask branch
class MaskBranch(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(MaskBranch, self).__init__()
        # the first residual block (RB)
        MB_RB1 = []
        MB_RB1.append(ResBlock(conv, n_feat, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1))
        # downscale (wait to implement)
        # 2 residual block in the middle place
        MB_RB2 = []
        for i in range(2):
            MB_RB2.append(ResBlock(conv, n_feat, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1))
        # upscale (wait to implement)   
        # 1 residual block after bottom-up top-down
        MB_RB3 = []
        MB_RB3.append(ResBlock(conv, n_feat, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1))
        # 1x1 conv layer
        MB_1x1conv = []
        MB_1x1conv.append(nn.Conv2d(n_feat,n_feat, 1, padding=0, bias=True))
        # sigmoid
        MB_sigmoid = []
        MB_sigmoid.append(nn.Sigmoid())

        self.MB_RB1 = nn.Sequential(*MB_RB1)
        self.MB_RB2 = nn.Sequential(*MB_RB2)
        self.MB_RB3 = nn.Sequential(*MB_RB3)
        self.MB_1x1conv = nn.Sequential(*MB_1x1conv)
        self.MB_sigmoid = nn.Sequential(*MB_sigmoid)
    
    def forward(self, x):
        x = self.MB_RB1(x)
        x = self.MB_RB2(x)
        x = self.MB_RB3(x)
        x = self.MB_1x1conv(x)
        mx = self.MB_sigmoid(x)

        return mx

## define residual attention module 
class ResAttModule(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(ResAttModule, self).__init__()
        RA_RB1 = []
        RA_RB1.append(ResBlock(conv, n_feat, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1))
        RA_TB = []
        RA_TB.append(TrunkBranch(conv, n_feat, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1))
        RA_MB = []
        RA_MB.append(MaskBranch(conv, n_feat, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1))
        RA_tail = []
        for i in range(2):
            RA_tail.append(ResBlock(conv, n_feat, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1))
        
        self.RA_RB1 = nn.Sequential(*RA_RB1)
        self.RA_TB  = nn.Sequential(*RA_TB)
        self.RA_MB  = nn.Sequential(*RA_MB)
        self.RA_tail = nn.Sequential(*RA_tail)

    def forward(self, x):
        x = self.RA_RB1(x)
        tx = self.RA_TB(x)
        mx = self.RA_MB(x)
        txmx = tx * mx
        hx = txmx + tx
        hx = self.RA_tail(hx)

        return hx

## define mask branch
class MaskBranchDownUp(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(MaskBranchDownUp, self).__init__()
        # the first residual block (RB)
        MB_RB1 = []
        MB_RB1.append(ResBlock(conv, n_feat, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1))
        # downscale (wait to implement)
        MB_Down = []
        MB_Down.append(nn.Conv2d(n_feat,n_feat, 3, stride=2, padding=1))
        # 2 residual block in the middle place
        MB_RB2 = []
        for i in range(2):
            MB_RB2.append(ResBlock(conv, n_feat, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1))
        # upscale (wait to implement)
        MB_Up = []
        MB_Up.append(nn.ConvTranspose2d(n_feat,n_feat, 6, stride=2, padding=2))   
        # 1 residual block after bottom-up top-down
        MB_RB3 = []
        MB_RB3.append(ResBlock(conv, n_feat, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1))
        # 1x1 conv layer
        MB_1x1conv = []
        MB_1x1conv.append(nn.Conv2d(n_feat,n_feat, 1, padding=0, bias=True))
        # sigmoid
        MB_sigmoid = []
        MB_sigmoid.append(nn.Sigmoid())

        self.MB_RB1 = nn.Sequential(*MB_RB1)
        self.MB_Down = nn.Sequential(*MB_Down)
        self.MB_RB2 = nn.Sequential(*MB_RB2)
        self.MB_Up  = nn.Sequential(*MB_Up)
        self.MB_RB3 = nn.Sequential(*MB_RB3)
        self.MB_1x1conv = nn.Sequential(*MB_1x1conv)
        self.MB_sigmoid = nn.Sequential(*MB_sigmoid)
    
    def forward(self, x):
        x_RB1 = self.MB_RB1(x)
        x_Down = self.MB_Down(x_RB1)
        x_RB2 = self.MB_RB2(x_Down)
        x_Up = self.MB_Up(x_RB2)
        x_preRB3 = x_RB1 + x_Up
        x_RB3 = self.MB_RB3(x_preRB3)
        x_1x1 = self.MB_1x1conv(x_RB3)
        mx = self.MB_sigmoid(x_1x1)

        return mx

## define nonlocal mask branch
class NLMaskBranchDownUp(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(NLMaskBranchDownUp, self).__init__()
        # the first residual block (RB)
        MB_RB1 = []
        # nonlocal block
        MB_RB1.append(NonLocalBlock2D(n_feat, n_feat // 2))
        MB_RB1.append(ResBlock(conv, n_feat, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1))
        # downscale (wait to implement)
        MB_Down = []
        MB_Down.append(nn.Conv2d(n_feat,n_feat, 3, stride=2, padding=1))
        # 2 residual block in the middle place
        MB_RB2 = []
        for i in range(2):
            MB_RB2.append(ResBlock(conv, n_feat, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1))
        # upscale (wait to implement)
        MB_Up = []
        MB_Up.append(nn.ConvTranspose2d(n_feat,n_feat, 6, stride=2, padding=2))   
        # 1 residual block after bottom-up top-down
        MB_RB3 = []
        MB_RB3.append(ResBlock(conv, n_feat, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1))
        # 1x1 conv layer
        MB_1x1conv = []
        MB_1x1conv.append(nn.Conv2d(n_feat,n_feat, 1, padding=0, bias=True))
        # sigmoid
        MB_sigmoid = []
        MB_sigmoid.append(nn.Sigmoid())

        self.MB_RB1 = nn.Sequential(*MB_RB1)
        self.MB_Down = nn.Sequential(*MB_Down)
        self.MB_RB2 = nn.Sequential(*MB_RB2)
        self.MB_Up  = nn.Sequential(*MB_Up)
        self.MB_RB3 = nn.Sequential(*MB_RB3)
        self.MB_1x1conv = nn.Sequential(*MB_1x1conv)
        self.MB_sigmoid = nn.Sequential(*MB_sigmoid)
    
    def forward(self, x):
        x_RB1 = self.MB_RB1(x)
        x_Down = self.MB_Down(x_RB1)
        x_RB2 = self.MB_RB2(x_Down)
        x_Up = self.MB_Up(x_RB2)
        x_preRB3 = x_RB1 + x_Up
        x_RB3 = self.MB_RB3(x_preRB3)
        x_1x1 = self.MB_1x1conv(x_RB3)
        mx = self.MB_sigmoid(x_1x1)

        return mx


## define residual attention module 
class ResAttModuleDownUp(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(ResAttModuleDownUp, self).__init__()
        RA_RB1 = []
        RA_RB1.append(ResBlock(conv, n_feat, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1))
        RA_TB = []
        RA_TB.append(TrunkBranch(conv, n_feat, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1))
        RA_MB = []
        RA_MB.append(MaskBranchDownUp(conv, n_feat, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1))
        RA_tail = []
        for i in range(2):
            RA_tail.append(ResBlock(conv, n_feat, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1))
        
        self.RA_RB1 = nn.Sequential(*RA_RB1)
        self.RA_TB  = nn.Sequential(*RA_TB)
        self.RA_MB  = nn.Sequential(*RA_MB)
        self.RA_tail = nn.Sequential(*RA_tail)

    def forward(self, x):
        x = self.RA_RB1(x)
        tx = self.RA_TB(x)
        mx = self.RA_MB(x)
        txmx = tx * mx
        hx = txmx + tx
        hx = self.RA_tail(hx)

        return hx

## define residual attention module plus
class ResAttModuleDownUpPlus(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(ResAttModuleDownUpPlus, self).__init__()
        RA_RB1 = []
        RA_RB1.append(ResBlock(conv, n_feat, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1))
        RA_TB = []
        RA_TB.append(TrunkBranch(conv, n_feat, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1))
        RA_MB = []
        RA_MB.append(MaskBranchDownUp(conv, n_feat, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1))
        RA_tail = []
        for i in range(2):
            RA_tail.append(ResBlock(conv, n_feat, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1))
        
        self.RA_RB1 = nn.Sequential(*RA_RB1)
        self.RA_TB  = nn.Sequential(*RA_TB)
        self.RA_MB  = nn.Sequential(*RA_MB)
        self.RA_tail = nn.Sequential(*RA_tail)

    def forward(self, input):
        RA_RB1_x = self.RA_RB1(input)
        tx = self.RA_TB(RA_RB1_x)
        mx = self.RA_MB(RA_RB1_x)
        txmx = tx * mx
        hx = txmx + RA_RB1_x
        hx = self.RA_tail(hx)

        return hx

## define nonlocal residual attention module 
class NLResAttModuleDownUp(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(NLResAttModuleDownUp, self).__init__()
        RA_RB1 = []
        RA_RB1.append(ResBlock(conv, n_feat, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1))
        RA_TB = []
        RA_TB.append(TrunkBranch(conv, n_feat, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1))
        RA_MB = []
        RA_MB.append(NLMaskBranchDownUp(conv, n_feat, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1))
        RA_tail = []
        for i in range(2):
            RA_tail.append(ResBlock(conv, n_feat, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1))
        
        self.RA_RB1 = nn.Sequential(*RA_RB1)
        self.RA_TB  = nn.Sequential(*RA_TB)
        self.RA_MB  = nn.Sequential(*RA_MB)
        self.RA_tail = nn.Sequential(*RA_tail)

    def forward(self, x):
        x = self.RA_RB1(x)
        tx = self.RA_TB(x)
        mx = self.RA_MB(x)
        txmx = tx * mx
        hx = txmx + tx
        hx = self.RA_tail(hx)

        return hx


## define nonlocal residual attention module plus
class NLResAttModuleDownUpPlus(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(NLResAttModuleDownUpPlus, self).__init__()
        RA_RB1 = []
        RA_RB1.append(ResBlock(conv, n_feat, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1))
        RA_TB = []
        RA_TB.append(TrunkBranch(conv, n_feat, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1))
        RA_MB = []
        RA_MB.append(NLMaskBranchDownUp(conv, n_feat, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1))
        RA_tail = []
        for i in range(2):
            RA_tail.append(ResBlock(conv, n_feat, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1))
        
        self.RA_RB1 = nn.Sequential(*RA_RB1)
        self.RA_TB  = nn.Sequential(*RA_TB)
        self.RA_MB  = nn.Sequential(*RA_MB)
        self.RA_tail = nn.Sequential(*RA_tail)

    def forward(self, input):
        RA_RB1_x = self.RA_RB1(input)
        tx = self.RA_TB(RA_RB1_x)
        mx = self.RA_MB(RA_RB1_x)
        txmx = tx * mx
        hx = txmx + RA_RB1_x
        hx = self.RA_tail(hx)

        return hx

## define residual attention module plus
class ResAttModuleDownUpPlus(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(ResAttModuleDownUpPlus, self).__init__()
        RA_RB1 = []
        RA_RB1.append(ResBlock(conv, n_feat, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1))
        RA_TB = []
        RA_TB.append(TrunkBranch(conv, n_feat, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1))
        RA_MB = []
        RA_MB.append(MaskBranchDownUp(conv, n_feat, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1))
        RA_tail = []
        for i in range(2):
            RA_tail.append(ResBlock(conv, n_feat, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1))
        
        self.RA_RB1 = nn.Sequential(*RA_RB1)
        self.RA_TB  = nn.Sequential(*RA_TB)
        self.RA_MB  = nn.Sequential(*RA_MB)
        self.RA_tail = nn.Sequential(*RA_tail)

    def forward(self, input):
        RA_RB1_x = self.RA_RB1(input)
        tx = self.RA_TB(RA_RB1_x)
        mx = self.RA_MB(RA_RB1_x)
        txmx = tx * mx
        hx = txmx + RA_RB1_x
        hx = self.RA_tail(hx)

        return hx## define residual attention module plus

# residual group, no mask
class ResGroup_Mask0(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(ResGroup_Mask0, self).__init__()
        RA_RB1 = []
        RA_RB1.append(ResBlock(conv, n_feat, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1))
        RA_TB = []
        RA_TB.append(TrunkBranch(conv, n_feat, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1))
        #RA_MB = []
        #RA_MB.append(MaskBranchDownUp(conv, n_feat, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1))
        RA_tail = []
        for i in range(2):
            RA_tail.append(ResBlock(conv, n_feat, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1))
        
        self.RA_RB1 = nn.Sequential(*RA_RB1)
        self.RA_TB  = nn.Sequential(*RA_TB)
        #self.RA_MB  = nn.Sequential(*RA_MB)
        self.RA_tail = nn.Sequential(*RA_tail)

    def forward(self, input):
        RA_RB1_x = self.RA_RB1(input)
        tx = self.RA_TB(RA_RB1_x)
        #mx = self.RA_MB(RA_RB1_x)
        #txmx = tx * mx
        #hx = txmx + RA_RB1_x
        hx = tx + RA_RB1_x
        hx = self.RA_tail(hx)

        return hx





