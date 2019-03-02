from model import common

import torch.nn as nn

def make_model(args, parent=False):
    return DNCA(args)

class DNCA(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(DNCA, self).__init__()

        n_resblock = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3 
        scale = args.scale[0]
        act = nn.ReLU(True)
        reduction = args.reduction 
        n_cab_1 = args.n_cab_1
        #n_cab_2 = args.n_cab_2
        n_cab_2 = 2 * n_cab_1

        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std)
        
        # define head module
        m_head = [conv(args.n_colors, n_feats, kernel_size)]
        # define CADown1
        m_CADown1 = [common.CADownLayer(n_feats, kernel_size, reduction)]
        # define n1 CAB
        m_n1_CAB_1 = [
            common.CABLayer(
                conv, n_feats, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(True)
            ) for _ in range(n_cab_1)
        ]
        # define CADown2
        m_CADown2 = [common.CADownLayer(n_feats, kernel_size, reduction)]
        # define n2 CAB
        m_n2_CAB = [
            common.CABLayer(
                conv, n_feats, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(True)
            ) for _ in range(n_cab_2)
        ]        
        # define CAUp1
        m_CAUp1 = [common.CAUpLayer(n_feats, kernel_size, reduction)]
        m_CAUp1.append(conv(n_feats, n_feats, kernel_size))
        
        # define n1 CAB
        m_n1_CAB_2 = [
            common.CABLayer(
                conv, n_feats, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(True)
            ) for _ in range(n_cab_1)
        ]         
        # define CAUp2
        m_CAUp2 = [common.CAUpLayer(n_feats, kernel_size, reduction)]
        m_CAUp2.append(conv(n_feats, n_feats, kernel_size))      
        # define tail module
        m_tail = [
            conv(n_feats, args.n_colors, kernel_size)
        ]

        self.add_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)
        
        self.head     = nn.Sequential(*m_head)
        self.CADown1  = nn.Sequential(*m_CADown1)
        self.n1_CAB_1 = nn.Sequential(*m_n1_CAB_1)
        self.CADown2  = nn.Sequential(*m_CADown2)
        self.n2_CAB   = nn.Sequential(*m_n2_CAB)
        self.CAUp1    = nn.Sequential(*m_CAUp1)
        self.n1_CAB_2 = nn.Sequential(*m_n1_CAB_2)
        self.CAUp2    = nn.Sequential(*m_CAUp2)
        self.tail     = nn.Sequential(*m_tail)

    def forward(self, input):
        #x = self.sub_mean(x)
        x1 = self.head(input)
        x2 = self.CADown1(x1)
        x2 = self.n1_CAB_1(x2)
        x3 = self.CADown2(x2)
        x3 = self.n2_CAB(x3)
        x4 = x2 + self.CAUp1(x3)
        x5 = self.n1_CAB_2(x4)
        x6 = x1 + self.CAUp2(x5)
        output = input + self.tail(x6)

        return output 

    def load_state_dict(self, state_dict, strict=False):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

