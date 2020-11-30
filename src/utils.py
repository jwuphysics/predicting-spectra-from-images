from fastai2.basics import *
from fastai2.vision.all import *
    
__all__ = ['root_mean_squared_error', 'replace_model_layer', 'GeM', 'gem', 'simple_cnn']    

# loss functions/metrics
def root_mean_squared_error(preds, targs): 
    return torch.sqrt(F.mse_loss(preds, targs))

# Generalized Mean
def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)

class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM,self).__init__()
        self.p = nn.Parameter(torch.ones(1)*p)
        self.eps = eps
    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)       
    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'

# for tweaking models
def replace_model_layer(model, layer_type_old, new_layer):
    conversion_count = 0
    for name, module in reversed(model._modules.items()):
        if len(list(module.children())) > 0:
            # recurse
            model._modules[name] = replace_model_layer(module, layer_type_old, new_layer)

        if type(module) == layer_type_old:
            layer_old = module
            model._modules[name] = new_layer

    return model


def simple_cnn(actns, kernel_szs=None, strides=None, bn=False, n_out=6) -> nn.Sequential:
    "CNN with `conv_layer` defined by `actns`, `kernel_szs` and `strides`, plus batchnorm if `bn`."
    nl = len(actns)-1
    kernel_szs = ifnone(kernel_szs, [3]*nl)
    strides    = ifnone(strides   , [2]*nl)
    layers = [ConvLayer(actns[i], actns[i+1], kernel_szs[i], stride=strides[i],
              norm_type=(NormType.Batch if bn and i<(len(strides)-1) else None)) for i in range_of(strides)]
    layers.append(PoolFlatten())
    layers.append(nn.Linear(actns[-1], n_out))
    return nn.Sequential(*layers)