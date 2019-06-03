import sys,os
import numpy as np
import torch
import torch.utils.data
from sklearn.utils import shuffle

EXTENSION='.pt'

########################################################################################################################

class DataSet(torch.utils.data.Dataset):

    def __init__(self,path_in,lchunk,stride,split='train',trim=None,
                 split_utterances=True,pc_split_utterances=0.1,split_speakers=False,pc_split_speakers=0.1,
                 frame_energy_thres=0,temp_jitter=False,
                 select_speaker=None,select_file=None,
                 seed=0,verbose=True,
                 store_in_ram=True,
                 sampling_rate=16000,do_audio_load=True,
                 ):
        self.path_in=path_in
        self.lchunk=lchunk
        self.stride=stride
        self.temp_jitter=temp_jitter
        self.store_in_ram=store_in_ram
        if trim is None or trim<=0:
            trim=np.inf

        # Get filenames in folder and subfolders
        self.filenames=[]
        for dirpath,dirnames,filenames in os.walk(self.path_in):
            for fn in filenames:
                if not fn.endswith(EXTENSION): continue
                new_fn=os.path.join(dirpath,fn)
                new_fn=os.path.relpath(new_fn,self.path_in)
                self.filenames.append(new_fn)
        self.filenames.sort()
        self.filenames=shuffle(self.filenames,random_state=seed)

        # Get speakers & utterances
        self.speakers={}
        self.utterances={}
        for fullfn in self.filenames:
            spk,ut=self.filename_split(fullfn)
            if spk not in self.speakers:
                self.speakers[spk]=len(self.speakers)
            if ut not in self.utterances:
                self.utterances[ut]=len(self.utterances)
        self.maxspeakers=len(self.speakers)

        # Split
        lutterances=list(self.utterances.keys())
        lutterances.sort()
        lutterances=shuffle(lutterances,random_state=seed)
        lspeakers=list(self.speakers.keys())
        lspeakers.sort()
        lspeakers=shuffle(lspeakers,random_state=seed)
        isplit_ut=int(len(lutterances)*pc_split_utterances)
        isplit_spk=int(len(lspeakers)*pc_split_speakers)
        if split=='train':
            spk_del=lspeakers[-2*isplit_spk:]
            ut_del=lutterances[-2*isplit_ut:]
        elif split=='valid':
            spk_del=lspeakers[:-2*isplit_spk]+lspeakers[-isplit_spk:]
            ut_del=lutterances[:-2*isplit_ut]+lutterances[-isplit_ut:]
        elif split=='train+valid':
            spk_del=lspeakers[-isplit_spk:]
            ut_del=lutterances[-isplit_ut:]
        elif split=='test':
            spk_del=lspeakers[:-isplit_spk]
            ut_del=lutterances[:-isplit_ut]
        else:
            print('Not implemented split',split)
            sys.exit()
        if split_speakers:
            for spk in spk_del:
                del self.speakers[spk]
        if split_utterances:
            for ut in ut_del:
                del self.utterances[ut]

        # Filter filenames by speaker and utterance
        filenames_new=[]
        for filename in self.filenames:
            spk,ut=self.filename_split(filename)
            if spk in self.speakers and ut in self.utterances:
                filenames_new.append(filename)
        self.filenames=filenames_new

        # Select speaker
        if select_speaker is not None:
            select_speaker=select_speaker.split(',')
            filenames_new=[]
            for filename in self.filenames:
                spk,ut=self.filename_split(filename)
                if spk in select_speaker and spk in self.speakers:
                    filenames_new.append(filename)
            if len(filenames_new)==0:
                print('\nERROR: Selected an invalid speaker. Options are:',list(self.speakers.keys()))
                sys.exit()
            self.filenames=filenames_new

        # Select specific file
        if select_file is not None:
            select_file=select_file.split(',')
            filenames_new=[]
            for filename in self.filenames:
                _,file=os.path.split(filename[:-len(EXTENSION)])
                if file in select_file:
                    filenames_new.append(filename)
            if len(filenames_new)==0:
                print('\nERROR: Selected an invalid file. Options are:',self.filenames[:int(np.min([50,len(self.filenames)]))],'... (without folder and without extension))')
                sys.exit()
            self.filenames=filenames_new

        # Indices!
        self.audios=[None]*len(self.filenames)
        self.indices=[]
        duration={}
        if do_audio_load:
            for i,filename in enumerate(self.filenames):
                if verbose:
                    print('\rRead audio {:5.1f}%'.format(100*(i+1)/len(self.filenames)),end='')
                # Info
                spk,ut=self.filename_split(filename)
                ispk,iut=self.speakers[spk],self.utterances[ut]
                # Load
                if spk not in duration:
                    duration[spk]=0
                if duration[spk]>=trim:
                    continue
                x=torch.load(os.path.join(self.path_in,filename))
                if self.store_in_ram:
                    self.audios[i]=x.clone()
                x=x.float()
                # Process
                for j in range(0,len(x),stride):
                    if j+self.lchunk>=len(x):
                        xx=x[j:]
                    else:
                        xx=x[j:j+self.lchunk]
                    if (xx.pow(2).sum()/self.lchunk).sqrt().item()>=frame_energy_thres:
                        info=[i,j,0,ispk,iut]
                        self.indices.append(torch.LongTensor(info))
                    duration[spk]+=stride/sampling_rate
                    if duration[spk]>=trim:
                        break
                self.indices[-1][2]=1
            if verbose:
                print()
            self.indices=torch.stack(self.indices)

        # Print
        if verbose:
            totalduration=0
            for key in duration.keys():
                totalduration+=duration[key]
            print('Loaded {:s}: {:.1f} h, {:d} spk, {:d} ut, {:d} files, {:d} frames (fet={:.1e},'.format(
                split,totalduration/3600,len(self.speakers),len(self.utterances),len(self.filenames),len(self.indices),frame_energy_thres),end='')
            if trim is None or trim>1e12:
                print(' no trim)')
            else:
                print(' trim={:.1f}s)'.format(trim))
            if select_speaker is not None:
                print('Selected speaker(s):',select_speaker)
            if select_file is not None:
                print('Selected file(s):',select_file)

        return

    def filename_split(self,fullfn):
        aux=os.path.split(fullfn)[-1][:-len(EXTENSION)].split('_')
        return aux[0],aux[1]

    def __len__(self):
        return self.indices.size(0)

    def __getitem__(self,idx):
        i,j,last,ispk,ichap=self.indices[idx,:]
        # Load file
        if self.store_in_ram:
            tmp=self.audios[i]
        else:
            tmp=torch.load(os.path.join(self.path_in,self.filenames[i]))
        # Temporal jitter
        if self.temp_jitter:
            j=j+np.random.randint(-self.stride//2,self.stride//2)
            if j<0:
                j=0
            elif j>len(tmp)-self.lchunk:
                j=np.max([0,len(tmp)-self.lchunk])
        # Get frame
        if j+self.lchunk>len(tmp):
            x=tmp[j:].float()
            x=torch.cat([x,torch.zeros(self.lchunk-len(x))])
        else:
            x=tmp[j:j+self.lchunk].float()
        # Get info
        y=torch.LongTensor([i,j,last,ispk,ichap])
        return x,y

########################################################################################################################

