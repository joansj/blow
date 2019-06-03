import sys,argparse,os,time
import numpy as np
import torch
import torch.utils.data
from copy import deepcopy

from utils import datain
from utils import utils
from utils import audio as audioutils

########################################################################################################################

# Arguments
parser=argparse.ArgumentParser(description='Audio synthesis script')
parser.add_argument('--seed_input',default=0,type=int,required=False,help='(default=%(default)d)')
parser.add_argument('--seed',default=0,type=int,required=False,help='(default=%(default)d)')
parser.add_argument('--device',default='cuda',type=str,required=False,help='(default=%(default)s)')
# Data
parser.add_argument('--trim',default=-1,type=float,required=False,help='(default=%(default)f)')
parser.add_argument('--base_fn_model',default='',type=str,required=True,help='(default=%(default)s)')
parser.add_argument('--path_out',default='../res/',type=str,required=True,help='(default=%(default)s)')
parser.add_argument('--split',default='test',type=str,required=False,help='(default=%(default)s)')
parser.add_argument('--force_source_file',default='',type=str,required=False,help='(default=%(default)s)')
parser.add_argument('--force_source_speaker',default='',type=str,required=False,help='(default=%(default)s)')
parser.add_argument('--force_target_speaker',default='',type=str,required=False,help='(default=%(default)s)')
# Conversion
parser.add_argument('--fn_list',default='',type=str,required=False,help='(default=%(default)s)')
parser.add_argument('--sbatch',default=256,type=int,required=False,help='(default=%(default)d)')
parser.add_argument('--convert',action='store_true')
parser.add_argument('--zavg',action='store_true',required=False,help='(default=%(default)s)')
parser.add_argument('--alpha',default=3,type=float,required=False,help='(default=%(default)f)')
# Synthesis
parser.add_argument('--lchunk',default=-1,type=int,required=False,help='(default=%(default)d)')
parser.add_argument('--stride',default=-1,type=int,required=False,help='(default=%(default)d)')
parser.add_argument('--synth_nonorm',action='store_true')
parser.add_argument('--maxfiles',default=10000000,type=int,required=False,help='(default=%(default)d)')

# Process arguments
args=parser.parse_args()
if args.trim<=0:
    args.trim=None
if args.force_source_file=='':
    args.force_source_file=None
if args.force_source_speaker=='':
    args.force_source_speaker=None
if args.force_target_speaker=='':
    args.force_target_speaker=None
if args.fn_list=='':
    args.fn_list='list_seed'+str(args.seed_input)+'_'+args.split+'.tsv'

# Print arguments
utils.print_arguments(args)

# Seed
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.device=='cuda':
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    torch.cuda.manual_seed(args.seed)

########################################################################################################################

# Load model, pars, & check

print('Load stuff')
report,pars,model,_=utils.load_stuff(args.base_fn_model)
traintimes,lens,_,losses=report
print('Training time = '+utils.timer(traintimes[0],traintimes[1]))
print('Training parameters =',pars)
print('Training set lengths =',lens)
try:
    losses_train,losses_valid,losses_test=np.vstack(losses['train']),np.vstack(losses['valid']),losses['test']
    print('Best losses = ',np.min(losses_train,axis=0),np.min(losses_valid,axis=0),losses_test)
except:
    print('[Could not load losses]')
print('-'*100)
model=model.to(args.device)
utils.print_model_report(model,verbose=1)
print('-'*100)

# Check lchunk & stride
if args.lchunk<=0:
    args.lchunk=pars.lchunk
if args.stride<=0:
    args.stride=args.lchunk//2
if pars.model!='blow' and not pars.model.startswith('test'):
    if not args.zavg:
        print('[WARNING: You are not using Blow. Are you sure you do not want to use --zavg?]')
    if args.lchunk!=pars.lchunk:
        args.lchunk=pars.lchunk
        print('[WARNING: '+pars.model+' model can only operate with same frame size as training. It has been changed, but you may want to change the stride now]')
if args.stride==args.lchunk:
    print('[Synth with 0% overlap]')
    window=torch.ones(args.lchunk)
elif args.stride==args.lchunk//2:
    print('[Synth with 50% overlap]')
    window=torch.hann_window(args.lchunk)
elif args.stride==args.lchunk//4*3:
    print('[Synth with 25% overlap]')
    window=torch.hann_window(args.lchunk//2)
    window=torch.cat([window[:len(window)//2],torch.ones(args.lchunk//2),window[len(window)//2:]])
else:
    print('[WARNING: No specific overlap strategy. Forcing Hann window and normalize]')
    window=torch.hann_window(args.lchunk)
    args.synth_nonorm=False
window=window.view(1,-1)

print('-'*100)

########################################################################################################################

def compute_speaker_averages(speakers,trim=5*60):
    print('('*100)
    print('[Averages]')
    select_speaker=None
    if args.force_source_speaker is not None and args.force_target_speaker is not None:
        select_speaker=args.force_source_speaker+','+args.force_target_speaker
    dataset=datain.DataSet(pars.path_data,args.lchunk,args.lchunk,sampling_rate=pars.sr,split='train+valid',trim=trim,
                           select_speaker=select_speaker,
                           seed=pars.seed)
    loader=torch.utils.data.DataLoader(dataset,batch_size=args.sbatch,shuffle=False,num_workers=0)
    averages={}
    count={}
    with torch.no_grad():
        for b,(x,idx) in enumerate(loader):
            x=x.to(args.device)
            s=idx[:,3].to(args.device)
            z=model.forward(x,s)[0]
            for n in range(len(idx)):
                i,j,last,ispk,ichap=idx[n]
                spk,_=dataset.filename_split(dataset.filenames[i])
                spk=speakers[spk]
                if spk not in averages:
                    averages[spk]=torch.zeros_like(z[n])
                    count[spk]=0
                averages[spk]+=z[n]
                count[spk]+=1
            print('\r---> Speaker(s) average: {:5.1f}%'.format(100*(b*args.sbatch+x.size(0))/len(dataset)),end='')
        print()
    for spk in averages.keys():
        averages[spk]=averages[spk]/count[spk]
    print(')'*100)
    return averages

########################################################################################################################

# Data
print('Load metadata')
dataset=datain.DataSet(pars.path_data,pars.lchunk,pars.stride,sampling_rate=pars.sr,split='train+valid',seed=pars.seed,do_audio_load=False)
speakers=deepcopy(dataset.speakers)
lspeakers=list(speakers.keys())
if args.zavg:
    averages=compute_speaker_averages(speakers)

# Input data
print('Load',args.split,'audio')
dataset=datain.DataSet(pars.path_data,args.lchunk,args.stride,sampling_rate=pars.sr,split=args.split,trim=args.trim,
                       select_speaker=args.force_source_speaker,select_file=args.force_source_file,
                       seed=pars.seed)
loader=torch.utils.data.DataLoader(dataset,batch_size=args.sbatch,shuffle=False,num_workers=0)

# Get transformation list
print('Transformation list')
np.random.seed(args.seed)
target_speaker=lspeakers[np.random.randint(len(lspeakers))]
if args.force_target_speaker is not None: target_speaker=args.force_target_speaker
fnlist=[]
itrafos=[]
nfiles=0
for x,info in loader:
    isource,itarget=[],[]
    for n in range(len(x)):

        # Get source and target speakers
        i,j,last,ispk,iut=info[n]
        source_speaker,_=dataset.filename_split(dataset.filenames[i])
        isource.append(speakers[source_speaker])
        itarget.append(speakers[target_speaker])
        if last==1 and nfiles<args.maxfiles:

            # Get filename
            fn=dataset.filenames[i][:-len(datain.EXTENSION)]
            fnlist.append([fn,source_speaker,target_speaker])

            # Restart
            target_speaker=lspeakers[np.random.randint(len(lspeakers))]
            if args.force_target_speaker is not None: target_speaker=args.force_target_speaker
            nfiles+=1

    isource,itarget=torch.LongTensor(isource),torch.LongTensor(itarget)
    itrafos.append([isource,itarget])
    if nfiles>=args.maxfiles:
        break

# Write transformation list
flist=open(os.path.join(args.path_out,args.fn_list),'w')
for fields in fnlist:
    flist.write('\t'.join(fields)+'\n')
flist.close()

########################################################################################################################

# Prepare model
try:
    model.precalc_matrices('on')
except:
    pass
model.eval()
print('-'*100)

# Synthesis loop
print('Synth')
audio=[]
nfiles=0
t_conv=0
t_synth=0
t_audio=0
try:
    with torch.no_grad():
        for k,(x,info) in enumerate(loader):
            if k>=len(itrafos):
                break
            isource,itarget=itrafos[k]

            # Track time
            tstart=time.time()

            # Convert
            if args.convert:
                # Forward & reverse
                x=x.to(args.device)
                isource=isource.to(args.device)
                itarget=itarget.to(args.device)
                z=model.forward(x,isource)[0]
                # Apply means?
                if args.zavg:
                    for n in range(len(x)):
                        z[n]=z[n]+args.alpha*(averages[itarget[n].item()]-averages[isource[n].item()])
                x=model.reverse(z,itarget)
                x=x.cpu()

            # Track time
            t_conv+=time.time()-tstart
            tstart=time.time()

            # Append audio
            x*=window
            for n in range(len(x)):
                audio.append(x[n])
                i,j,last,ispk,iut=info[n]
                if last==1:

                    # Filename
                    fn,source_speaker,target_speaker=fnlist[nfiles]
                    _,fn=os.path.split(fn)
                    if args.convert:
                        fn+='_to_'+target_speaker
                    fn=os.path.join(args.path_out,fn+'.wav')

                    # Synthesize
                    print(str(nfiles+1)+'/'+str(len(fnlist))+'\t'+fn)
                    sys.stdout.flush()
                    audioutils.synthesize(audio,fn,args.stride,sr=pars.sr,normalize=not args.synth_nonorm)

                    # Track time
                    t_audio+=((len(audio)-1)*args.stride+args.lchunk)/pars.sr

                    # Reset
                    audio=[]
                    nfiles+=1
                    if nfiles>=args.maxfiles:
                        break

            # Track time
            t_synth+=time.time()-tstart
except KeyboardInterrupt:
    print()

########################################################################################################################

# Report times
print('-'*100)
print('Time')
print('   Conversion:\t{:6.1f} ms/s'.format(1000*t_conv/t_audio))
print('   Synthesis:\t{:6.1f} ms/s'.format(1000*t_synth/t_audio))
print('   TOTAL:\t{:6.1f} ms/s\t(x{:.1f})'.format(1000*(t_conv+t_synth)/t_audio,1/((t_conv+t_synth)/t_audio)))
print('-'*100)

# Done
if args.convert:
    print('*** Conversions done ***')
else:
    print('*** Original audio. No conversions done ***')
