
from fastai2.vision.all import *
from deconvolution.models.deconv import FastDeconv # from `deconvolution` repo

class DeconvLayer(nn.Sequential):
    "Create a sequence of deconv (`ni` to `nf`) and ReLU/Mish (if `use_activ`) layers."
    def __init__(self, ni, nf, ks=3, stride=1, padding=None, bias=True, ndim=2,
                 act_cls=defaults.activation, transpose=False, init='auto', xtra=None, bias_std=0.01, **kwargs):
        if padding is None: padding = ((ks-1)//2 if not transpose else 0)
        conv_func = FastDeconv
        conv = conv_func(ni, nf, kernel_size=ks, bias=bias, stride=stride, padding=padding, **kwargs)
        act = None if act_cls is None else act_cls()
        init_linear(conv, act, init=init, bias_std=bias_std)
        layers = [conv]
        if act: layers.append(act)
        if xtra: layers.append(xtra)
        super().__init__(*layers)
        
        
class XResNet_hybrid(nn.Sequential):
    """An xresnet-like architecture with deconv layers in the stem and Conv2d elsewhere."""
    @delegates(ResBlock)
    def __init__(self, block, expansion, layers, p=0.0, c_in=3, n_out=1000, stem_szs=(32,32,64),
                 widen=1.0, sa=False, act_cls=defaults.activation, **kwargs):
        store_attr(self, 'block,expansion,act_cls')
        stem_szs = [c_in, *stem_szs]
        stem = [DeconvLayer(stem_szs[i], stem_szs[i+1], stride=2 if i==0 else 1, act_cls=act_cls)
                for i in range(3)]

        block_szs = [int(o*widen) for o in [64,128,256,512] +[256]*(len(layers)-4)]
        block_szs = [64//expansion] + block_szs
        blocks    = self._make_blocks(layers, block_szs, sa, **kwargs)

        super().__init__(
            *stem, nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            *blocks,
            nn.AdaptiveAvgPool2d(1), Flatten(), nn.Dropout(p),
            nn.Linear(block_szs[-1]*expansion, n_out),
        )
        init_cnn(self)

    def _make_blocks(self, layers, block_szs, sa, **kwargs):
        return [self._make_layer(ni=block_szs[i], nf=block_szs[i+1], blocks=l,
                                 stride=1 if i==0 else 2, sa=sa and i==len(layers)-4, **kwargs)
                for i,l in enumerate(layers)]

    def _make_layer(self, ni, nf, blocks, stride, sa, **kwargs):
        return nn.Sequential(
            *[self.block(self.expansion, ni if i==0 else nf, nf, stride=stride if i==0 else 1,
                      sa=sa and i==(blocks-1), act_cls=self.act_cls, **kwargs)
              for i in range(blocks)])
    
def _xresnet_hybrid(pretrained, expansion, layers, **kwargs):
    return XResNet_hybrid(ResBlock, expansion, layers, **kwargs)

def xresnet18_hybrid (pretrained=False, **kwargs): return _xresnet_hybrid(pretrained, 1, [2, 2,  2, 2], **kwargs)
def xresnet34_hybrid (pretrained=False, **kwargs): return _xresnet_hybrid(pretrained, 1, [3, 4,  6, 3], **kwargs)
def xresnet50_hybrid (pretrained=False, **kwargs): return _xresnet_hybrid(pretrained, 4, [3, 4,  6, 3], **kwargs)

# going to full Deconvolution Networks

class ResBlock_deconv(Module):
    "Resnet block from `ni` to `nh` with `stride`"
    @delegates(DeconvLayer.__init__)
    def __init__(self, expansion, ni, nf, stride=1, groups=1, reduction=None, nh1=None, nh2=None, dw=False, g2=1,
                 sa=False, sym=False, act_cls=defaults.activation, ndim=2, ks=3,
                 pool=AvgPool, pool_first=True, **kwargs):
        if nh2 is None: nh2 = nf
        if nh1 is None: nh1 = nh2
        nf,ni = nf*expansion,ni*expansion
        k0 = dict(act_cls=act_cls, ndim=ndim, **kwargs)
        k1 = dict(act_cls=None, ndim=ndim, **kwargs)
        convpath  = [DeconvLayer(ni,  nh2, ks, stride=stride, groups=ni if dw else groups, **k0),
                     DeconvLayer(nh2,  nf, ks, groups=g2, **k1)
        ] if expansion == 1 else [
                     DeconvLayer(ni,  nh1, 1, **k0),
                     DeconvLayer(nh1, nh2, ks, stride=stride, groups=nh1 if dw else groups, **k0),
                     DeconvLayer(nh2,  nf, 1, groups=g2, **k1)]
        if reduction: convpath.append(SEModule(nf, reduction=reduction, act_cls=act_cls))
        if sa: convpath.append(SimpleSelfAttention(nf,ks=1,sym=sym))
        self.convpath = nn.Sequential(*convpath)
        idpath = []
        if ni!=nf: idpath.append(DeconvLayer(ni, nf, 1, act_cls=None, ndim=ndim, **kwargs))
        if stride!=1: idpath.insert((1,0)[pool_first], pool(stride, ndim=ndim, ceil_mode=True))
        self.idpath = nn.Sequential(*idpath)
        self.act = defaults.activation(inplace=True) if act_cls is defaults.activation else act_cls()

    def forward(self, x): return self.act(self.convpath(x) + self.idpath(x))
    
class XResNet_deconv(nn.Sequential):
    """An xresnet-like architecture with deconv layers throughout."""
    @delegates(ResBlock_deconv)
    def __init__(self, block, expansion, layers, p=0.0, c_in=3, n_out=1000, stem_szs=(32,32,64),
                 widen=1.0, sa=False, act_cls=defaults.activation, **kwargs):
        store_attr(self, 'block,expansion,act_cls')
        stem_szs = [c_in, *stem_szs]
        stem = [DeconvLayer(stem_szs[i], stem_szs[i+1], stride=2 if i==0 else 1, act_cls=act_cls)
                for i in range(3)]

        block_szs = [int(o*widen) for o in [64,128,256,512] +[256]*(len(layers)-4)]
        block_szs = [64//expansion] + block_szs
        blocks    = self._make_blocks(layers, block_szs, sa, **kwargs)

        super().__init__(
            *stem, nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            *blocks,
            nn.AdaptiveAvgPool2d(1), Flatten(), nn.Dropout(p),
            nn.Linear(block_szs[-1]*expansion, n_out),
        )
        init_cnn(self)

    def _make_blocks(self, layers, block_szs, sa, **kwargs):
        return [self._make_layer(ni=block_szs[i], nf=block_szs[i+1], blocks=l,
                                 stride=1 if i==0 else 2, sa=sa and i==len(layers)-4, **kwargs)
                for i,l in enumerate(layers)]

    def _make_layer(self, ni, nf, blocks, stride, sa, **kwargs):
        return nn.Sequential(
            *[self.block(self.expansion, ni if i==0 else nf, nf, stride=stride if i==0 else 1,
                      sa=sa and i==(blocks-1), act_cls=self.act_cls, **kwargs)
              for i in range(blocks)])
    
def _xresnet_deconv(pretrained, expansion, layers, **kwargs):
    return XResNet_deconv(ResBlock_deconv, expansion, layers, **kwargs)

def xresnet18_deconv (pretrained=False, **kwargs): return _xresnet_deconv(pretrained, 1, [2, 2,  2, 2], **kwargs)
def xresnet34_deconv (pretrained=False, **kwargs): return _xresnet_deconv(pretrained, 1, [3, 4,  6, 3], **kwargs)
def xresnet50_deconv (pretrained=False, **kwargs): return _xresnet_deconv(pretrained, 4, [3, 4,  6, 3], **kwargs)
