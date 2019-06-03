import sys
import torch

from utils import audio

########################################################################################################################
########################################################################################################################

class Model(torch.nn.Module):

    def __init__(self,sqfactor,nblocks,nflows,ncha,ntargets,_,semb=128):
        super(Model,self).__init__()

        nsq=sqfactor
        print('Channels/squeeze = ',end='')
        self.blocks=torch.nn.ModuleList()
        for _ in range(nblocks):
            self.blocks.append(Block(sqfactor,nflows,nsq,ncha,semb))
            print('{:d}, '.format(nsq),end='')
            nsq*=sqfactor
        print()
        self.final_nsq=nsq//sqfactor

        self.embedding=torch.nn.Embedding(ntargets,semb)

        return

    def forward(self,h,s):
        # Prepare
        sbatch,lchunk=h.size()
        h=h.unsqueeze(1)
        emb=self.embedding(s)
        # Run blocks & accumulate log-det
        log_det=0
        for block in self.blocks:
            h,ldet=block.forward(h,emb)
            log_det+=ldet
        # Back to original dim
        h=h.view(sbatch,lchunk)
        return h,log_det

    def reverse(self,h,s):
        # Prepare
        sbatch,lchunk=h.size()
        h=h.view(sbatch,self.final_nsq,lchunk//self.final_nsq)
        emb=self.embedding(s)
        # Run blocks
        for block in self.blocks[::-1]:
            h=block.reverse(h,emb)
        # Back to original dim
        h=h.squeeze(1)
        # Postproc
        h=audio.proc_problematic_samples(h)
        return h

    def precalc_matrices(self,mode):
        if mode!='on' and mode!='off':
            print('[precalc_matrices() needs either on or off]')
            sys.exit()
        if mode=='off':
            for i in range(len(self.blocks)):
                for j in range(len(self.blocks[i].flows)):
                    self.blocks[i].flows[j].mixer.weight=None
                    self.blocks[i].flows[j].mixer.invweight=None
        else:
            for i in range(len(self.blocks)):
                for j in range(len(self.blocks[i].flows)):
                    self.blocks[i].flows[j].mixer.weight=self.blocks[i].flows[j].mixer.calc_weight()
                    self.blocks[i].flows[j].mixer.invweight=self.blocks[i].flows[j].mixer.weight.inverse()
        return

########################################################################################################################

class Block(torch.nn.Module):

    def __init__(self,sqfactor,nflows,nsq,ncha,semb):
        super(Block,self).__init__()

        self.squeeze=Squeezer(factor=sqfactor)
        self.flows=torch.nn.ModuleList()
        for _ in range(nflows):
            self.flows.append(Flow(nsq,ncha,semb))

        return

    def forward(self,h,emb):
        # Squeeze
        h=self.squeeze.forward(h)
        # Run flows & accumulate log-det
        log_det=0
        for flow in self.flows:
            h,ldet=flow.forward(h,emb)
            log_det+=ldet
        return h,log_det

    def reverse(self,h,emb):
        # Run flows
        for flow in self.flows[::-1]:
            h=flow.reverse(h,emb)
        # Unsqueeze
        h=self.squeeze.reverse(h)
        return h

########################################################################################################################

class Flow(torch.nn.Module):

    def __init__(self,nsq,ncha,semb):
        super(Flow,self).__init__()

        self.mixer=InvConv(nsq)
        self.norm=ActNorm(nsq)
        self.coupling=AffineCoupling(nsq,ncha,semb)

        return

    def forward(self,h,emb):
        logdet=0
        h,ld=self.mixer.forward(h)
        logdet=logdet+ld
        h,ld=self.norm.forward(h)
        logdet=logdet+ld
        h,ld=self.coupling.forward(h,emb)
        logdet=logdet+ld
        return h,logdet

    def reverse(self,h,emb):
        h=self.coupling.reverse(h,emb)
        h=self.norm.reverse(h)
        h=self.mixer.reverse(h)
        return h

########################################################################################################################
########################################################################################################################

class Squeezer(object):
    def __init__(self,factor=2):
        self.factor=factor
        return

    def forward(self,h):
        sbatch,nsq,lchunk=h.size()
        h=h.view(sbatch,nsq,lchunk//self.factor,self.factor)
        h=h.permute(0,1,3,2).contiguous()
        h=h.view(sbatch,nsq*self.factor,lchunk//self.factor)
        return h

    def reverse(self,h):
        sbatch,nsq,lchunk=h.size()
        h=h.view(sbatch,nsq//self.factor,self.factor,lchunk)
        h=h.permute(0,1,3,2).contiguous()
        h=h.view(sbatch,nsq//self.factor,lchunk*self.factor)
        return h

########################################################################################################################

from scipy import linalg
import numpy as np

class InvConv(torch.nn.Module):

    def __init__(self,in_channel):
        super(InvConv,self).__init__()

        weight=np.random.randn(in_channel,in_channel)
        q,_=linalg.qr(weight)
        w_p,w_l,w_u=linalg.lu(q.astype(np.float32))
        w_s=np.diag(w_u)
        w_u=np.triu(w_u,1)
        u_mask=np.triu(np.ones_like(w_u),1)
        l_mask=u_mask.T

        self.register_buffer('w_p',torch.from_numpy(w_p))
        self.register_buffer('u_mask',torch.from_numpy(u_mask))
        self.register_buffer('l_mask',torch.from_numpy(l_mask))
        self.register_buffer('l_eye',torch.eye(l_mask.shape[0]))
        self.register_buffer('s_sign',torch.sign(torch.from_numpy(w_s)))
        self.w_l=torch.nn.Parameter(torch.from_numpy(w_l))
        self.w_s=torch.nn.Parameter(torch.log(1e-7+torch.abs(torch.from_numpy(w_s))))
        self.w_u=torch.nn.Parameter(torch.from_numpy(w_u))

        self.weight=None
        self.invweight=None

        return

    def calc_weight(self):
        weight=(
            self.w_p
            @ (self.w_l*self.l_mask+self.l_eye)
            @ (self.w_u*self.u_mask+torch.diag(self.s_sign*(torch.exp(self.w_s))))
        )
        return weight

    def forward(self,h):
        if self.weight is None:
            weight=self.calc_weight()
        else:
            weight=self.weight
        h=torch.nn.functional.conv1d(h,weight.unsqueeze(2))
        logdet=self.w_s.sum()*h.size(2)
        return h,logdet

    def reverse(self,h):
        if self.invweight is None:
            invweight=self.calc_weight().inverse()
        else:
            invweight=self.invweight
        h=torch.nn.functional.conv1d(h,invweight.unsqueeze(2))
        return h

########################################################################################################################

class ActNorm(torch.nn.Module):
    def __init__(self,nsq,data_init=True):
        super(ActNorm,self).__init__()
        self.initialized=not data_init

        self.m=torch.nn.Parameter(torch.zeros(1,nsq,1))
        self.logs=torch.nn.Parameter(torch.zeros(1,nsq,1))

        return

    def forward(self,h):
        # Init
        if not self.initialized:
            sbatch,nsq,lchunk=h.size()
            flatten=h.permute(1,0,2).contiguous().view(nsq,-1).data
            self.m.data=-flatten.mean(1).view(1,nsq,1)
            self.logs.data=torch.log(1/(flatten.std(1)+1e-7)).view(1,nsq,1)
            self.initialized=True
        # Normalize
        h=torch.exp(self.logs)*(h+self.m)
        logdet=self.logs.sum()*h.size(2)
        return h,logdet

    def reverse(self,h):
        return h*torch.exp(-self.logs)-self.m

########################################################################################################################

class AffineCoupling(torch.nn.Module):

    def __init__(self,nsq,ncha,semb):
        super(AffineCoupling,self).__init__()
        self.net=CouplingNet(nsq//2,ncha,semb)
        return

    def forward(self,h,emb):
        h1,h2=torch.chunk(h,2,dim=1)
        s,m=self.net.forward(h1,emb)
        h2=s*(h2+m)
        h=torch.cat([h1,h2],1)
        logdet=s.log().sum(2).sum(1)
        return h,logdet

    def reverse(self,h,emb):
        h1,h2=torch.chunk(h,2,dim=1)
        s,m=self.net.forward(h1,emb)
        h2=h2/s-m
        h=torch.cat([h1,h2],1)
        return h


class CouplingNet(torch.nn.Module):

    def __init__(self,nsq,ncha,semb,kw=3):
        super(CouplingNet,self).__init__()
        assert kw%2==1
        assert ncha%nsq==0
        self.ncha=ncha
        self.kw=kw
        self.adapt_w=torch.nn.Linear(semb,ncha*kw)
        self.adapt_b=torch.nn.Linear(semb,ncha)
        self.net=torch.nn.Sequential(
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv1d(ncha,ncha,1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv1d(ncha,2*nsq,kw,padding=kw//2),
        )
        self.net[-1].weight.data.zero_()
        self.net[-1].bias.data.zero_()
        return

    def forward(self,h,emb):
        sbatch,nsq,lchunk=h.size()
        h=h.contiguous()
        """
        # Slower version
        ws=list(self.adapt_w(emb).view(sbatch,self.ncha,1,self.kw))
        bs=list(self.adapt_b(emb))
        hs=list(torch.chunk(h,sbatch,dim=0))
        out=[]
        for hi,wi,bi in zip(hs,ws,bs):
            out.append(torch.nn.functional.conv1d(hi,wi,bias=bi,padding=self.kw//2,groups=nsq))
        h=torch.cat(out,dim=0)
        """
        # Faster version fully using group convolution
        w=self.adapt_w(emb).view(-1,1,self.kw)
        b=self.adapt_b(emb).view(-1)
        h=torch.nn.functional.conv1d(h.view(1,-1,lchunk),w,bias=b,padding=self.kw//2,groups=sbatch*nsq).view(sbatch,self.ncha,lchunk)
        #"""
        h=self.net.forward(h)
        s,m=torch.chunk(h,2,dim=1)
        s=torch.sigmoid(s+2)+1e-7
        return s,m

########################################################################################################################
########################################################################################################################
