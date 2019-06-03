import sys,os,argparse
import torch

# Arguments
parser=argparse.ArgumentParser(description='Renaming script (runs over preprocessed files)')
parser.add_argument('--dataset',type=str,required=True,help='(default=%(default)s)',
                    choices=['vctk','librispeech','nsynthp'])
parser.add_argument('--path',default='',type=str,required=True,help='(default=%(default)s)')
parser.add_argument('--extension',default='.pt',type=str,required=False,help='(default=%(default)s)')
parser.add_argument('--execute',action='store_true')

args=parser.parse_args()

if args.execute:
    input('Are you sure you want to rename? ')

# Get all filenames
all_fn=[]
for dirpath,dirnames,filenames in os.walk(args.path):
    for filename in filenames:
        if filename.endswith(args.extension):
            fn=os.path.join(dirpath,filename)
            all_fn.append(fn)

# Process datasets
if args.dataset=='vctk':

    # ----- VCTK -----

    # Load fn2ut
    utterances,fn2ut=torch.load(os.path.join('utils','vctk_utterance_files_map.pt'))
    i=0
    for ut in utterances:
        utterances[ut]=i
        i+=1
    # Loop files
    for i,fn_old in enumerate(all_fn):
        folder,fn=os.path.split(fn_old[:-len(args.extension)])
        spk=fn.split('_')[0]
        try:
            ut='{:05d}'.format(utterances[fn2ut[fn]])
            fn_new=os.path.join(folder,spk+'_'+ut+args.extension)
            print(i+1,'\t'+fn_old+' ---> '+fn_new)
            if args.execute:
                os.rename(fn_old,fn_new)
        except:
            print('Not found correspondence for '+fn_old+' - Deleting')
            if args.execute:
                os.remove(fn_old)
            else:
                input()

elif args.dataset=='librispeech':

    # --- LibriSpeech ---

    # Loop files
    for i,fn_old in enumerate(all_fn):
        folder,fn=os.path.split(fn_old[:-len(args.extension)])
        fields=fn.split('_')
        spk,chap=fields[0],fields[0]+'-'+fields[1]
        fn_new=os.path.join(folder,spk+'_'+chap+'_'+'-'.join(fields[2:])+args.extension)
        print(i+1,'\t'+fn_old+' ---> '+fn_new)
        if args.execute:
            os.rename(fn_old,fn_new)

elif args.dataset=='nsynthp':

    # --- NSynth (pitch) ---

    # Loop files
    for i,fn_old in enumerate(all_fn):
        folder,fn=os.path.split(fn_old[:-len(args.extension)])
        instr,pitch,vel=fn.split('-')
        instr=instr.replace('_','')
        fn_new=os.path.join(folder,pitch+'_'+instr+'-'+vel+args.extension)
        print(i+1,'\t'+fn_old+' ---> '+fn_new)
        if args.execute:
            os.rename(fn_old,fn_new)


# *** Add other options here ***

