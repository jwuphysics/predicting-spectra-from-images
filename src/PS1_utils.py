from fastai2.basics import *
from fastai2.vision.all import *


__all__ = ['grizyTensorImage', 'grizyImageBlock', 'grizyCrop', 'norm', 'open_npy', 'show_composite', 'show_decoded_results'] + [f'vae_{i}' for i in range(1, 7)]

vmin, vmax = -0.541, 3.907

def open_npy(fn):
    return torch.from_numpy(np.nan_to_num(np.load(fn))).float()
           
def show_composite(img, ax=None, figsize=(3,3), title=None, scale=True,
                   ctx=None, vmin=None, vmax=None, **kwargs)->plt.Axes:
    "Show three channel composite so that channels correspond to grizy"
    ax = ifnone(ax, ctx)
    if ax is None: _, ax = plt.subplots(figsize=figsize)    
    tempim = img.data.cpu().numpy()
    im = np.zeros((tempim.shape[1], tempim.shape[2], 3))
    im[...,0] = np.mean([tempim[0], 0.5*tempim[1]], axis=0) 
    im[...,1] = np.mean([0.5*tempim[1], tempim[2]], axis=0)
    im[...,2] = np.mean([tempim[3], tempim[4]], axis=0)
    if scale: im = norm(im, vmin, vmax)
    ax.imshow(np.clip(im, 0, 1), **kwargs)
    ax.axis('off')
    if title is not None: ax.set_title(title)
    return ax


def norm(vals, vmin=None, vmax=None, Q=8, stretch=None):
    """
    For visualization purposes normalize image with `arcsinh((vals-vmin)/(vmax-vmin)), 
    with vals either specified or within 0.01 and 0.99 quantiles of all values. 
    
    Q and stretch control the arcsinh softening parameter, see Lupton et al. 2004 and
    see https://docs.astropy.org/en/stable/_modules/astropy/visualization/lupton_rgb.html#make_lupton_rgb
    """
    vmin = ifnone(vmin, np.quantile(vals, 0.01))
    vmax = ifnone(vmax, np.quantile(vals, 0.99))
    
    if stretch is None:
        return np.arcsinh(Q*(vals - vmin) / (vmax-vmin)) / Q
    else:
        return np.arcsinh(Q*(vals - vmin) / stretch) / Q

class grizyTensorImage(TensorImage):
    _show_args = ArrayImageBase._show_args
    def show(self, ctx=None, vmin=None, vmax=None, **kwargs):
        return show_composite(self, ctx=ctx, vmin=vmin, vmax=vmax, **{**self._show_args, **kwargs})

    @classmethod
    def create(cls, fn,  **kwargs) ->None:
        if str(fn).endswith('.npy'): return cls(open_npy(fn=fn))
        
    def __repr__(self): return f'{self.__class__.__name__} size={"x".join([str(d) for d in self.shape])}'
    
grizyTensorImage.create = Transform(grizyTensorImage.create) 

def grizyImageBlock(): 
    return TransformBlock(partial(grizyTensorImage.create))

# augmentation transforms
class grizyCrop(Transform):
    """A square center crop
    """
    order=0
    def __init__(self, size):
        super().__init__()
        self.size = size
    def encodes(self, o: grizyTensorImage):
        orig_sz = o.shape[-1]
        tl = (orig_sz-self.size)//2
        o = o[..., tl:-tl, tl:-tl]
        return o

# VAE utils
def _get_VAE_model():
    sys.path.append('/home/jupyter/morphological-spectra/SDSS-VAE')
    VAE = torch.load(f'/home/jupyter/morphological-spectra/SDSS-VAE/64k_20190612/0057.pth').eval()
    return VAE

def _get_latent_PCA_model():
    with open(f'/home/jupyter/morphological-spectra/SDSS-VAE/latent_pca.pkl', 'rb') as f:
        latent_PCA = pickle.load(f)
    return latent_PCA

def _get_meanspec():
    return np.load('/home/jupyter/morphological-spectra/SDSS-VAE/meanspec.npy')

def show_decoded_results(dls, cnn_model, vmin=vmin, vmax=vmax, Q=8,
                         latent_PCA=None, VAE=None, meanspec=None, 
                         ncols=4, nrows=1, dpi=100, scale=True):
    
    assert dls.bs > ncols*nrows, "You're requesting more plots than can fit in a single batch!"
    x, y = dls.one_batch()    
    p = cnn_model(x)
    
    latent_PCA = ifnone(latent_PCA, _get_latent_PCA_model())
    VAE = ifnone(VAE, _get_VAE_model())
    meanspec = ifnone(meanspec, _get_meanspec())
    
    x, y, p = x.detach(), y.detach(), p.detach()

    idxs = np.random.choice(range(dls.bs), size=nrows*ncols, replace=False)

    # figure
    fig = plt.figure(figsize=(5*ncols, 6*nrows), dpi=dpi)
    gs = matplotlib.gridspec.GridSpec(2*nrows, ncols, height_ratios=nrows*[0.8, 5]) 


    for i, idx in enumerate(idxs):
        with torch.no_grad():
            p_recon = VAE.decode(torch.tensor(latent_PCA.inverse_transform(p[idx].cpu()), dtype=torch.float32)).numpy() + meanspec
            y_recon = VAE.decode(torch.tensor(latent_PCA.inverse_transform(y[idx].cpu()), dtype=torch.float32)).numpy() + meanspec
        
        spec_idx = i + (ncols) * (i//ncols)
        im_idx = i+ncols + (ncols)*(i//ncols)
        
        ax_spec = plt.subplot(gs[spec_idx])
        ax_spec.plot(y_recon, c='k', lw=1)
        ax_spec.plot(p_recon, c='C3')
        ax_spec.set_ylim(*np.quantile(p_recon, [0.01, 0.99]))
        ax_spec.axis('off')

        ax_im = plt.subplot(gs[im_idx])
        show_composite(x[idx], ax=ax_im, vmin=vmin, vmax=vmax)
        
    return fig, fig.get_axes()

# useful metrics
def vae_1(input, target): return torch.sqrt(torch.pow(input - target, 2).mean(0)[0])
def vae_2(input, target): return torch.sqrt(torch.pow(input - target, 2).mean(0)[1])
def vae_3(input, target): return torch.sqrt(torch.pow(input - target, 2).mean(0)[2])
def vae_4(input, target): return torch.sqrt(torch.pow(input - target, 2).mean(0)[3])
def vae_5(input, target): return torch.sqrt(torch.pow(input - target, 2).mean(0)[4])
def vae_6(input, target): return torch.sqrt(torch.pow(input - target, 2).mean(0)[5])