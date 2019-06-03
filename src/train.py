import sys,argparse,time,os
import numpy as np
import torch
import torch.utils.data

from utils import datain
from utils import utils
from utils import audio as audioutils

########################################################################################################################

# Arguments
parser=argparse.ArgumentParser(description='Training script')
parser.add_argument('--seed',default=0,type=int,required=False,help='(default=%(default)d)')
parser.add_argument('--device',default='cuda',type=str,required=False,help='(default=%(default)s)')
parser.add_argument('--nworkers',default=0,type=int,required=False,help='(default=%(default)d)')
# --- Data
parser.add_argument('--path_data',default='',type=str,required=True,help='(default=%(default)s)')
parser.add_argument('--sr',default=16000,type=int,required=False,help='(default=%(default)d)')
parser.add_argument('--trim',default=-1,type=float,required=False,help='(default=%(default)f)')
parser.add_argument('--lchunk',default=4096,type=int,required=False,help='(default=%(default)d)')
parser.add_argument('--stride',default=-1,type=int,required=False,help='(default=%(default)d)')
parser.add_argument('--frame_energy_thres',default=0.025,type=float,required=False,help='(default=%(default)f)')
parser.add_argument('--augment',default=1,type=int,required=False,help='(default=%(default)d)')
# --- Optimization & training
parser.add_argument('--nepochs',default=999,type=int,required=False,help='(default=%(default)d)')
parser.add_argument('--sbatch',default=38,type=int,required=False,help='(default=%(default)d)')
parser.add_argument('--optim',default='adam',type=str,required=False,help='(default=%(default)s',
                     choices=['adam','sgd','sgdm','adabound'])
parser.add_argument('--lr',default=1e-4,type=float,required=False,help='(default=%(default)f)')
parser.add_argument('--lr_thres',default=1e-4,type=float,required=False,help='(default=%(default)f)')
parser.add_argument('--lr_patience',default=10,type=int,required=False,help='(default=%(default)d)')
parser.add_argument('--lr_factor',default=0.2,type=float,required=False,help='(default=%(default)f)')
parser.add_argument('--lr_restarts',default=2,type=int,required=False,help='(default=%(default)d)')
parser.add_argument('--multigpu',action='store_true')
# --- Model
parser.add_argument('--model',type=str,required=True,help='(default=%(default)s)',
                    choices=['realnvp','glow','glow_wn','blow','blow2','test1','test2','test3'])
parser.add_argument('--load_existing',default='',type=str,required=False,help='(default=%(default)s)')
parser.add_argument('--nsqueeze',default=2,type=int,required=False,help='(default=%(default)d)')
parser.add_argument('--nblocks',default=8,type=int,required=False,help='(default=%(default)d)')
parser.add_argument('--nflows',default=12,type=int,required=False,help='(default=%(default)d)')
parser.add_argument('--ncha',default=512,type=int,required=False,help='(default=%(default)d)')
# --- Results
parser.add_argument('--base_fn_out',default='',type=str,required=True,help='(default=%(default)s)')

# Process arguments
args=parser.parse_args()
if args.trim<=0:
    args.trim=None
if args.stride<=0:
    args.stride=args.lchunk
if args.multigpu:
    args.ngpus=torch.cuda.device_count()
    args.sbatch*=args.ngpus
else:
    args.ngpus=1
utils.print_arguments(args)

# Seed
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.device=='cuda':
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    torch.cuda.manual_seed(args.seed)

########################################################################################################################

# Revive previous experiment?
if args.load_existing!='':
    print('Load previous experiment')
    report,pars,model,_=utils.load_stuff(args.load_existing)
    print('[Loaded model]')
    utils.print_model_report(model,verbose=1)
    #utils.print_arguments(pars)
    _,_,loss_best,losses=report
    print('[Loaded report with {:d} epochs; best validation was {:.2f}]'.format(len(losses['valid']),loss_best))

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device=='cuda': torch.cuda.manual_seed(args.seed)

    pars.load_existing=args.load_existing
    pars.base_fn_out=args.base_fn_out
    pars.multigpu=args.multigpu
    pars.sbatch=args.sbatch
    pars.optim=args.optim
    pars.lr=args.lr
    args=pars
    print('New arguments')
    utils.print_arguments(args)

########################################################################################################################

# Data sets and data loaders
print('Data')
dataset_train=datain.DataSet(args.path_data,args.lchunk,args.stride,split='train',sampling_rate=args.sr,
                             trim=args.trim,frame_energy_thres=args.frame_energy_thres,
                             temp_jitter=args.augment>0,
                             seed=args.seed)
dataset_valid=datain.DataSet(args.path_data,args.lchunk,args.stride,split='valid',sampling_rate=args.sr,
                             trim=args.trim,frame_energy_thres=args.frame_energy_thres,
                             temp_jitter=False,
                             seed=args.seed)
dataset_test=datain.DataSet(args.path_data,args.lchunk,args.stride,split='test',sampling_rate=args.sr,
                            trim=args.trim,frame_energy_thres=args.frame_energy_thres,
                            temp_jitter=False,
                            seed=args.seed)
loader_train=torch.utils.data.DataLoader(dataset_train,batch_size=args.sbatch,shuffle=True,drop_last=True,num_workers=args.nworkers)
loader_valid=torch.utils.data.DataLoader(dataset_valid,batch_size=args.sbatch,shuffle=False,num_workers=args.nworkers)
loader_test=torch.utils.data.DataLoader(dataset_test,batch_size=args.sbatch,shuffle=False,num_workers=args.nworkers)
print('-'*100)

########################################################################################################################

batch_data_augmentation=audioutils.DataAugmentation(args.device)

def batch_loop(args,e,eval,dataset,loader,msg_pre='',exit_at_first_fwd=False):
    # Prepare
    if eval:
        model.eval()
    else:
        model.train()

    # Loop
    cum_losses=0
    cum_num=0
    msg_post=''
    for b,(x,info) in enumerate(loader):

        # Prepare data
        s=info[:,3].to(args.device)
        x=x.to(args.device)
        if not eval and args.augment>0:
            if args.augment>1:
                x=batch_data_augmentation.noiseg(x,0.001)
            x=batch_data_augmentation.emphasis(x,0.2)
            x=batch_data_augmentation.magnorm_flip(x,1)
            if args.augment>1:
                x=batch_data_augmentation.compress(x,0.1)

        # Forward
        z,log_det=model.forward(x,s)
        loss,losses=utils.loss_flow_nll(z,log_det)

        """
        # Test reverse
        if e==0 and b==5:
            with torch.no_grad():
                model.eval()
                z,_=model.forward(x,s)
                xhat=model.reverse(z,s)
            dif=(x-xhat).abs()
            print()
            print(z[0].view(-1).cpu())
            print(xhat[0].view(-1).cpu())
            print('AvgDif =',dif.mean().item(),' MaxDif =',dif.max().item())
            print(dif.view(-1).cpu())
            sys.exit()
        #"""

        # Exit?
        if exit_at_first_fwd:
            return loss,losses,msg_pre

        # Backward
        if not eval:
            optim.zero_grad()
            loss.backward()
            optim.step()

        # Report/print
        cum_losses+=losses*len(x)
        cum_num+=len(x)
        msg='\r| T = '+utils.timer(tstart,time.time())+' | '
        msg+='Epoch = {:3d} ({:5.1f}%) | '.format(e+1,100*(b*args.sbatch+len(info))/len(dataset))
        if eval: msg_post='Eval loss = '
        else: msg_post='Train loss = '
        for i in range(len(cum_losses)):
            msg_post+='{:7.2f} '.format(cum_losses[i]/cum_num)
        msg_post+='| '
        print(msg+msg_pre+msg_post,end='')

    cum_losses/=cum_num
    return cum_losses[0],cum_losses,msg_pre+msg_post

########################################################################################################################

def make_report():
    report=[
        (tstart,time.time()),
        [len(dataset_train),len(dataset_valid),len(dataset_test)],
        loss_best,
        losses,
    ]
    return report

def load_best_model():
    _,_,model,_=utils.load_stuff(args.base_fn_out)
    model=model.to(args.device)
    if args.multigpu:
        model=torch.nn.DataParallel(model)
    return model

def get_optimizer(lr):
    if args.optim=='adam':
        args.clearmomentum=True
        return torch.optim.Adam(model.parameters(),lr=lr)
    elif args.optim=='sgd':
        args.clearmomentum=True
        return torch.optim.SGD(model.parameters(),lr=lr)
    elif args.optim=='sgdm':
        args.clearmomentum=False
        return torch.optim.SGD(model.parameters(),lr=lr,momentum=0.85)
    elif args.optim=='adabound':
        import adabound
        args.clearmomentum=False
        return adabound.AdaBound(model.parameters(),lr=lr)
    return None

########################################################################################################################

# Init
print('Init')
tstart=time.time()
utils.save_stuff(args.base_fn_out,args=args)

# New experiment?
if args.load_existing=='':
    losses=None
    # Get model
    print('New {:s} model'.format(args.model))
    if args.model=='realnvp':
        from models import realnvp as sel_model
    elif args.model=='glow_wn':
        from models import glow_wn as sel_model
    elif args.model=='glow':
        from models import glow as sel_model
    elif args.model=='blow':
        from models import blow as sel_model
    elif args.model=='blow2':
        from models import blow2 as sel_model
    elif args.model=='test1':
        from models import test1 as sel_model
    elif args.model=='test2':
        from models import test2 as sel_model
    elif args.model=='test3':
        from models import test3 as sel_model
    model=sel_model.Model(args.nsqueeze,args.nblocks,args.nflows,args.ncha,dataset_train.maxspeakers,args.lchunk)
    utils.print_model_report(model,verbose=1)

model.to(args.device)
optim=get_optimizer(args.lr)

# If new model, init it
if args.load_existing=='':
    losses={'train':[],'valid':[],'test':np.inf}
    loss_best=np.inf
    print('Forward init')
    with torch.no_grad():
        batch_loop(args,-1,False,dataset_train,loader_train,exit_at_first_fwd=True)

# Placeholder save
utils.save_stuff(args.base_fn_out,report=make_report(),model=model)

# Multigpu
if args.multigpu:
    print('[Using {:d} GPUs]'.format(torch.cuda.device_count()))
    model=torch.nn.DataParallel(model)

print('-'*100)

# Train
print('Train')
lr=args.lr
patience=args.lr_patience
restarts=args.lr_restarts
try:
    for e in range(args.nepochs):

        # Run
        _,losses_train,msg=batch_loop(args,e,False,dataset_train,loader_train)
        losses['train'].append(losses_train)
        with torch.no_grad():
            loss,losses_valid,_=batch_loop(args,e,True,dataset_valid,loader_valid,msg_pre=msg)
            losses['valid'].append(losses_valid)

        # Control stall
        if np.isnan(loss) or loss>1000:
            patience=0
            loss=np.inf
            model=load_best_model()

        # Best model?
        if loss<loss_best*(1+args.lr_thres):
            print('*',end='')
            loss_best=loss
            patience=args.lr_patience
            utils.save_stuff(args.base_fn_out,report=make_report(),model=model)
        else:
            # Learning rate annealing or exit
            patience-=1
            if patience<=0:
                restarts-=1
                if restarts<0:
                    print('End')
                    break
                lr*=args.lr_factor
                print('lr={:.1e}'.format(lr),end='')
                if args.clearmomentum:
                    optim=get_optimizer(lr)
                else:
                    for pg in optim.param_groups:
                        pg['lr']=lr
                patience=args.lr_patience

        print()
except KeyboardInterrupt:
    print()
print('-'*100)

# Test
print('Test')
model=load_best_model()
try:
    with torch.no_grad():
        _,losses['test'],_=batch_loop(args,-2,True,dataset_test,loader_test)
    utils.save_stuff(args.base_fn_out,report=make_report())
except KeyboardInterrupt:
    pass
print()
print('-'*100)

########################################################################################################################

# Done
print('Done')
