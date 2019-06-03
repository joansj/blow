import sys
import torch
import numpy as np
from scipy.io import wavfile
import numba

########################################################################################################################

class DataAugmentation(object):

    def __init__(self,device,betaparam=0.2):
        self.device=device
        self.betadist=torch.distributions.beta.Beta(betaparam,betaparam)
        return

    def _get_random_vector(self,size):
        if self.device=='cuda':
            return torch.cuda.FloatTensor(size,1).uniform_()
        return torch.rand(size,1)

    def magnorm(self,x,val):
        return x*val*(self._get_random_vector(len(x))/(x.abs().max(1,keepdim=True)[0]+1e-7))

    def flip(self,x):
        return x*(self._get_random_vector(len(x))-0.5).sign()

    def magnorm_flip(self,x,val):
        return x*val*((2*self._get_random_vector(len(x))-1)/(x.abs().max(1,keepdim=True)[0]+1e-7))

    def compress(self,x,val):
        return x.sign()*(x.abs()**(1+val*(2*self._get_random_vector(len(x))-1)))

    def noiseu(self,x,val):
        return x+val*self._get_random_vector(len(x))*(2*torch.rand_like(x)-1)

    def noiseg(self,x,val):
        return x+val*self._get_random_vector(len(x))*torch.randn_like(x)

    def emphasis(self,x,val):
        # http://www.fon.hum.uva.nl/praat/manual/Sound__Filter__de-emphasis____.html
        # (unrolled and truncated version; performs normalization but might be better to re-normalize afterwards)
        alpha=val*(2*self._get_random_vector(len(x))-1)
        sign=alpha.sign()
        alpha=alpha.abs()
        xorig=x.clone(); mx=torch.ones_like(alpha)
        x[:,1:]+=sign*alpha*xorig[:,:-1]; mx+=alpha; alpha*=alpha
        x[:,2:]+=sign*alpha*xorig[:,:-2]; mx+=alpha; alpha*=alpha
        x[:,3:]+=sign*alpha*xorig[:,:-3]; mx+=alpha; alpha*=alpha
        x[:,4:]+=sign*alpha*xorig[:,:-4]; mx+=alpha#; alpha*=alpha
        return x/mx

    def mixup(self,x):
        lamb=self.betadist.sample((len(x),1)).to(x.device)
        lamb=torch.max(lamb,1-lamb)
        perm=torch.randperm(len(x)).to(x.device)
        return lamb*x+(1-lamb)*x[perm]

########################################################################################################################

def proc_problematic_samples(x,soft=True):
    x[torch.isnan(x)]=0
    if soft:
        x=softclamp(x)
    else:
        x=torch.clamp(x,-1,1)
    return x

def softclamp(x,mx=1,margin=0.03,alpha=0.7,clipval=100):
    x=torch.clamp(x,-clipval,clipval)
    xabs=x.abs()
    rmargin=mx-margin
    mask=(xabs<rmargin).float()
    x=mask*x+(1-mask)*torch.sign(x)*((1-torch.exp(-alpha*(xabs-rmargin)/margin))*margin+rmargin)
    return x

########################################################################################################################

def synthesize(frames,filename,stride,sr=16000,deemph=0,ymax=0.98,normalize=False):
    # Generate stream
    y=torch.zeros((len(frames)-1)*stride+len(frames[0]))
    for i,x in enumerate(frames):
        y[i*stride:i*stride+len(x)]+=x
    # To numpy & deemph
    y=y.numpy().astype(np.float32)
    if deemph>0:
        y=deemphasis(y,alpha=deemph)
    # Normalize
    if normalize:
        y-=np.mean(y)
        mx=np.max(np.abs(y))
        if mx>0:
            y*=ymax/mx
    else:
        y=np.clip(y,-ymax,ymax)
    # To 16 bit & save
    wavfile.write(filename,sr,np.array(y*32767,dtype=np.int16))
    return y

########################################################################################################################

@numba.jit(nopython=True,cache=True)
def deemphasis(x,alpha=0.2):
    # http://www.fon.hum.uva.nl/praat/manual/Sound__Filter__de-emphasis____.html
    assert 0<=alpha<=1
    if alpha==0 or alpha==1:
        return x
    y=x.copy()
    for n in range(1,len(x)):
        y[n]=x[n]+alpha*y[n-1]
    return y

########################################################################################################################
