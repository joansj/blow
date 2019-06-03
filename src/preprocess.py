import sys,argparse,os,time
import librosa,pysptk
import numpy as np
import torch

from utils import utils

# Arguments
parser=argparse.ArgumentParser(description='Preprocessing script')
parser.add_argument('--path_in',default='',type=str,required=True,help='(default=%(default)s)')
parser.add_argument('--extension',default='.xxx',type=str,required=True,help='(default=%(default)s)')
parser.add_argument('--path_out',default='',type=str,required=True,help='(default=%(default)s)')
parser.add_argument('--sr',default=16000,type=int,required=False,help='(default=%(default)s)')
parser.add_argument('--nonormalize',action='store_true')
parser.add_argument('--maxmag',default=0.99,type=float,required=False,help='(default=%(default)s)')

args=parser.parse_args()
args.path_in=args.path_in.split(',')
for i in range(len(args.path_in)):
    args.path_in[i]=os.path.normpath(args.path_in[i])
args.path_out=os.path.normpath(args.path_out)

print('='*100)
print('Arguments =')
for arg in vars(args):
    print('\t'+arg+':',getattr(args,arg))
print('='*100)

########################################################################################################################

tstart=time.time()

if os.path.exists(args.path_out):
    print('Delete destination path '+args.path_out)
    os.system('rm -rf '+args.path_out)
print('Create destination path '+args.path_out)
os.system('mkdir '+args.path_out)

print('Loop files')
n=0
for path_in in args.path_in:
    print(path_in)
    for dirpath,dirnames,filenames in os.walk(path_in):
        newpath=dirpath.replace(path_in,args.path_out)
        if not os.path.exists(newpath):
            os.system('mkdir '+newpath)
        for fn in filenames:
            if not fn.endswith(args.extension): continue
            n+=1
            fullfn_in=os.path.join(dirpath,fn)
            fullfn_out=fullfn_in.replace(path_in,args.path_out).replace(args.extension,'.pt')
            print(str(n)+'\t'+utils.timer(tstart,time.time())+'\t'+fullfn_in+' --> '+fullfn_out)

            x,_=librosa.load(fullfn_in,sr=args.sr)

            if not args.nonormalize:
                x-=np.mean(x)
                x*=args.maxmag/(np.max(np.abs(x))+1e-7)

            torch.save(torch.HalfTensor(x),fullfn_out)

print('Done')

print('[Do you need to run misc/rename_dataset.py? Check the readme]')

########################################################################################################################
