import sys,argparse,os,subprocess
import numpy as np

# Arguments
parser=argparse.ArgumentParser(description='Audio listening script')
parser.add_argument('--path_refs_train',default='',type=str,required=True,help='(default=%(default)s)')
parser.add_argument('--path_refs_test',default='',type=str,required=True,help='(default=%(default)s)')
parser.add_argument('--paths_convs',default='',type=str,required=True,help='(default=%(default)s)')
parser.add_argument('--player',default='',type=str,required=True,help='(default=%(default)s)')
parser.add_argument('--extension',default='.wav',type=str,required=False,help='(default=%(default)s)')
parser.add_argument('--delimiters',default='_to_,-vcto-',type=str,required=False,help='(default=%(default)s)')
parser.add_argument('--force_file',default='',type=str,required=False,help='(default=%(default)s)')
args=parser.parse_args()

args.paths_convs=args.paths_convs.split(',')
args.delimiters=args.delimiters.split(',')
if args.force_file=='':
    args.force_file=None

########################################################################################################################

print('='*100)
print('Load references...')

# Load train refs
fn_refs_train=[]
for dirpath,dirnames,filenames in os.walk(args.path_refs_train):
    for fn in filenames:
        if fn.endswith(args.extension):
            fn_refs_train.append(os.path.join(dirpath,fn))
print(args.path_refs_train,':',len(fn_refs_train),'references (train)')

# Load test refs
fn_refs_test=[]
for dirpath,dirnames,filenames in os.walk(args.path_refs_test):
    for fn in filenames:
        if fn.endswith(args.extension):
            fn_refs_test.append(os.path.join(dirpath,fn))
print(args.path_refs_test,':',len(fn_refs_test),'references (test)')

# Load conversions
print('Load conversions...')
fn_conv={}
convmin=np.inf
pathmin=None
for path in args.paths_convs:
    fn_conv[path]=[]
    for dirpath,dirnames,filenames in os.walk(path):
        for fn in filenames:
            if fn.endswith(args.extension):
                if args.force_file is None or args.force_file in fn:
                    fn_out=os.path.join(dirpath,fn)
                    spk=None
                    for sep in args.delimiters:
                        if sep in fn:
                            fn,spk=fn.split(sep)
                            break
                    if spk is None:
                        continue
                    spk_ref=spk[:-len(args.extension)]
                    fn_in=os.path.join(args.path_refs_test,fn+args.extension)
                    fn_conv[path].append([fn_in,spk_ref,fn_out])
    print(path,':',len(fn_conv[path]),'conversions')
    if len(fn_conv[path])<convmin:
        convmin=len(fn_conv[path])
        pathmin=path

print('='*100)

########################################################################################################################

# Play
print('Running test...')
answers=[]
exit=False
np.random.shuffle(fn_conv[pathmin])
n=0
for fn_in,spk_ref,fn_out in fn_conv[pathmin]:
    print('-'*100)
    np.random.shuffle(fn_refs_train)
    for fn in fn_refs_train:
        if spk_ref+'_' in fn:
            fn_ref=fn
            break
    print('R:',fn_ref,'-->',fn_in)

    tests={}
    for path in fn_conv.keys():
        for new_fn_in,new_spk_ref,new_fn_out in fn_conv[path]:
            if new_fn_in==fn_in and new_spk_ref==spk_ref:
                tests[path]=new_fn_out
                break
    order=list(tests.keys())
    np.random.shuffle(order)

    nvotes=0
    while True:
        key=input('Q{:d}: [s/t/1-{:d}/v1-v{:d}/q/n] '.format(n+1,len(tests),len(tests))).lower()
        if key=='q':
            exit=True
            break
        elif key=='n':
            break
        elif key=='s':
            subprocess.call([args.player,fn_in],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        elif key=='t':
            subprocess.call([args.player,fn_ref],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        elif key.isdigit():
            num=int(key)-1
            subprocess.call([args.player,tests[order[num]]],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        elif key[0]=='v':
            nvotes+=1
            print('Voted for system '+key[1]+'! ({:d} votes)'.format(nvotes))
            num=int(key[1])-1
            answers.append(order[num])
        else:
            continue
    n+=1
    if exit:
        break

########################################################################################################################

print('='*100)
print('Vote count:')
count={}
for path in fn_conv.keys():
    count[path]=0
for ans in answers:
    count[ans]+=1
for path in count.keys():
    print(path,'\t',count[path])
print('='*100)
