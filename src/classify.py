import sys,argparse,os
import librosa
from sklearn import preprocessing
from sklearn.utils import shuffle
import torch
import numpy as np

from utils import utils

# Arguments
parser=argparse.ArgumentParser(description='')
parser.add_argument('--device',default='cuda',type=str,required=False,help='(default=%(default)s)')
parser.add_argument('--mode',type=str,required=True,help='(default=%(default)s)',
                    choices=['train','valid','test','random'])
parser.add_argument('--path_in',default='',type=str,required=True,help='(default=%(default)s)')
parser.add_argument('--fn_mask',default='_to_',type=str,required=False,help='(default=%(default)s)')
parser.add_argument('--fn_cla',default='',type=str,required=True,help='(default=%(default)s)')
parser.add_argument('--fn_res',default='',type=str,required=True,help='(default=%(default)s)')
parser.add_argument('--extension',default='.wav',type=str,required=False,help='(default=%(default)s)')
parser.add_argument('--pc_valid',default=0.1,type=float,required=False,help='(default=%(default)f)')
parser.add_argument('--patience',default=5,type=int,required=False,help='(default=%(default)f)')
parser.add_argument('--cut_last',default=0,type=int,required=False,help='(default=%(default)f)')
args=parser.parse_args()

utils.print_arguments(args)

########################################################################################################################

speakers={}
if args.mode!='train':
    print('Load classifier')
    speakers,sca,model=torch.load(args.fn_cla)

########################################################################################################################

# Load filenames
print('Filenames')
fn_all=[]
for dirpath,dirnames,filenames in os.walk(args.path_in):
    for fn in filenames:
        if fn.endswith(args.extension):
            if args.mode!='test' or (args.mode=='test' and args.fn_mask in fn):
                fn_all.append(os.path.join(dirpath,fn))
print(len(fn_all))

# Feature extraction
print('Feature extraction')
durations=[]
features=[]
labels=[]
source,target=[],[]
try:
    for i,fn in enumerate(fn_all):
        arrs=[]

        y,sr=librosa.load(fn,sr=16000)
        if args.cut_last>0:
            y=y[:-args.cut_last-1]
        spec=np.abs(librosa.stft(y=y,n_fft=2048,hop_length=128,win_length=256))**2
        melspec=librosa.feature.melspectrogram(S=spec,n_mels=200)
        mfcc=librosa.feature.mfcc(S=librosa.power_to_db(melspec),n_mfcc=40)
        arrs.append(mfcc)
        mfcc=librosa.feature.delta(mfcc)
        arrs.append(mfcc)
        mfcc=librosa.feature.delta(mfcc)
        arrs.append(mfcc)
        #cqt=librosa.amplitude_to_db(np.abs(librosa.cqt(y,sr=sr,fmin=27.5,n_bins=96)),ref=np.max)
        #arrs.append(cqt)
        #cqt=librosa.feature.delta(cqt)
        #arrs.append(cqt)
        rms=librosa.feature.rms(y=y)
        arrs.append(rms)
        #zcr=librosa.feature.zero_crossing_rate(y=y)
        #arrs.append(zcr)

        feat=[]
        for x in arrs:
            feat+=list(np.mean(x,axis=1))
            feat+=list(np.std(x,axis=1))

        spk=os.path.split(fn)[-1].split('_')[0]
        if args.mode=='train' and spk not in speakers:
            speakers[spk]=len(speakers)
        elif spk not in speakers:
            continue
        source.append(speakers[spk])
        if args.mode=='test' or args.mode=='random':
            spk=os.path.split(fn)[-1].split(args.fn_mask)[-1][:-len(args.extension)]
            if spk not in speakers:
                continue
        target.append(speakers[spk])
        yy=(np.abs(y)>0.02).astype(np.float32)

        durations.append(np.sum(yy)/sr)
        features.append(feat)
        labels.append(speakers[spk])

        print('\r{:5.1f}%'.format(100*(i+1)/len(fn_all)),end='')
except KeyboardInterrupt:
    pass
print()
durations=np.array(durations,dtype=np.float32)
features=np.array(features,dtype=np.float32)
labels=np.array(labels,dtype=np.int32)
source=np.array(source,dtype=np.int32)
if target[0] is not None:
    target=np.array(target,dtype=np.int32)
print(len(speakers),'speakers')
print(features.shape,labels.shape)

########################################################################################################################

def batch_loop(e,r,x,y,eval):
    if eval:
        model.eval()
    else:
        model.train()
        r=shuffle(r)
    losses=[]
    predictions=[]
    for b in range(0,len(r),sbatch):
        if b+sbatch>len(r):
            rr=r[b:]
        else:
            rr=r[b:b+sbatch]
        rr=torch.LongTensor(rr)
        xb=x[rr,:].to(args.device)
        yb=y[rr].to(args.device)
        ybhat=model.forward(xb)
        loss=loss_function(ybhat,yb)
        losses+=list(loss.data.cpu().numpy())
        predictions+=list(ybhat.data.max(1)[1].cpu().numpy())
        if not eval:
            loss=loss.mean()
            optim.zero_grad()
            loss.backward()
            optim.step()
        print('\rEpoch {:03d}/{:03d} - {:5.1f}% : loss = {:7.3f}'.format(e+1,nepochs,100*len(losses)/len(x),np.mean(losses)),end='')
    return losses,predictions

print('-'*100)
nepochs=200
sbatch=128
loss_function=torch.nn.CrossEntropyLoss(reduction='none')

if args.mode=='train':
    print('Train')
    sca=preprocessing.StandardScaler()
    x=sca.fit_transform(features)
    x,y=torch.FloatTensor(x),torch.LongTensor(labels)
    model=torch.nn.Sequential(torch.nn.Dropout(0.4),torch.nn.Linear(x.size(1),len(speakers)))
    model=model.to(args.device)
    optim=torch.optim.Adam(model.parameters())
    r=list(range(len(x)))
    r=shuffle(r)
    split=int(args.pc_valid*len(r))
    r_train,r_valid=r[:-split],r[-split:]
    try:
        loss_best=np.inf
        patience=args.patience
        for e in range(nepochs):
            batch_loop(e,r_train,x,y,False)
            with torch.no_grad():
                losses,_=batch_loop(e,r_valid,x,y,True)
            print()
            if np.mean(losses)<loss_best:
                loss_best=np.mean(losses)
                patience=args.patience
            else:
                patience-=1
                if patience==0:
                    break
    except KeyboardInterrupt:
        print()
    torch.save([speakers,sca,model.cpu()],args.fn_cla)
    print('[Saved '+args.fn_cla+']')

    print('Predict')
    x,y=x[r_valid,:],y[r_valid]

else:
    print('Predict')
    x=sca.transform(features)
    x,y=torch.FloatTensor(x),torch.LongTensor(labels)

if args.mode=='random':
    losses,predictions=[],[]
    ymax=np.max(y.numpy())
    for i in range(len(y)):
        losses.append(0)
        predictions.append(np.random.randint(ymax+1))
else:
    model=model.to(args.device)
    with torch.no_grad():
        losses,predictions=batch_loop(-1,list(range(len(x))),x,y,True)
        print()
losses=np.array(losses,dtype=np.float32)
predictions=np.array(predictions,dtype=np.int32)
print('NLL = {:7.3f}'.format(np.mean(losses)))
print('Accuracy = {:5.1f}%'.format(100*np.mean((predictions==y.numpy()).astype(np.float32))))
print('-'*100)

########################################################################################################################

torch.save([durations,losses,predictions,labels,speakers,source,target],args.fn_res)
print('[Saved '+args.fn_res+']')
