from model import common

import torch.nn as nn

def make_model(args, parent=False):
    return RNLRANSR(args)
### RNLRANSR
### residual attention + downscale upscale + super-resolution
class _ResGroup(nn.Module):
    def __init__(self, conv, n_feats, kernel_size, act, res_scale):
        super(_ResGroup, self).__init__()
        modules_body = []
        #modules_body.append(common.NonLocalBlock2D(n_feats, n_feats // 2))
        modules_body.append(common.ResAttModuleDownUpPlus(conv, n_feats, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1))
        # if we don't use group residual, donot remove the following conv
        modules_body.append(conv(n_feats, n_feats, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        #res += x
        return res

### nonlocal residual attention + downscale upscale + super-resolution
class _NLResGroup(nn.Module):
    def __init__(self, conv, n_feats, kernel_size, act, res_scale):
        super(_NLResGroup, self).__init__()
        modules_body = []
        #modules_body.append(common.NonLocalBlock2D(n_feats, n_feats // 2))
        modules_body.append(common.NLResAttModuleDownUpPlus(conv, n_feats, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1))
        # if we don't use group residual, donot remove the following conv
        modules_body.append(conv(n_feats, n_feats, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        #res += x
        return res

class RNLRANSR(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(RNLRANSR, self).__init__()
        
        n_resgroup = args.n_resgroups
        n_resblock = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3
        reduction = args.reduction 
        scale = args.scale[0]
        act = nn.ReLU(True)
        
        # RGB mean for DIV2K 1-800
        #rgb_mean = (0.4488, 0.4371, 0.4040)
        # RGB mean for DIVFlickr2K 1-3450
        # rgb_mean = (0.4690, 0.4490, 0.4036)
        
        if args.data_train == 'DIV2K':
            print('Use DIV2K mean (0.4488, 0.4371, 0.4040)')
            rgb_mean = (0.4488, 0.4371, 0.4040)
        elif args.data_train == 'DIVFlickr2K':
            print('Use DIVFlickr2K mean (0.4690, 0.4490, 0.4036)')
            rgb_mean = (0.4690, 0.4490, 0.4036)
        rgb_std = (1.0, 1.0, 1.0)    
        self.sub_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std)
        
        # define head module
        modules_head = [conv(args.n_colors, n_feats, kernel_size)]
        #modules_head.append(common.NonLocalBlock2D(n_feats, n_feats // 2))
        # define body module
        '''
        modules_body = [
            common.ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=args.res_scale) \
            for _ in range(n_resblock)]
        '''
        modules_body_nl_low = [
            _NLResGroup(
                conv, n_feats, kernel_size, act=act, res_scale=args.res_scale)]
        modules_body = [
            _ResGroup(
                conv, n_feats, kernel_size, act=act, res_scale=args.res_scale) \
            for _ in range(n_resgroup - 2)]
        modules_body_nl_high = [
            _NLResGroup(
                conv, n_feats, kernel_size, act=act, res_scale=args.res_scale)]
        modules_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        
        modules_tail = [
            common.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, args.n_colors, kernel_size)]
        '''
        modules_tail = [
            conv(n_feats, args.n_colors, kernel_size)]
        '''
        self.add_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)

        self.head = nn.Sequential(*modules_head)
        self.body_nl_low = nn.Sequential(*modules_body_nl_low)
        self.body = nn.Sequential(*modules_body)
        self.body_nl_high = nn.Sequential(*modules_body_nl_high)
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x):
        #if self.args.shift_mean:
        #    x = self.sub_mean(x)
        x = self.sub_mean(x)
        feats_shallow = self.head(x)

        res = self.body_nl_low(feats_shallow)
        res = self.body(res)
        res = self.body_nl_high(res)
        res += feats_shallow

        res_main = self.tail(res)

        #res_clean = x + res_main
        #if self.args.shift_mean:
        #    x = self.add_mean(x)
        res_main = self.add_mean(res_main)

        #return res_clean
        return res_main  

    def load_state_dict(self, state_dict, strict=False):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') >= 0:
                        print('Replace pre-trained upsampler to new one...')
                    else:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError('missing keys in state_dict: "{}"'.format(missing))

