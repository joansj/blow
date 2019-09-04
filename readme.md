# Blow: a single-scale hyperconditioned flow for non-parallel raw-audio voice conversion

## Info

### Abstract

End-to-end models for raw audio generation are a challenge, specially if they have to work with non-parallel data, which is a desirable setup in many situations. Voice conversion, in which a model has to impersonate a speaker in a recording, is one of those situations. In this paper, we propose Blow, a single-scale normalizing flow using hypernetwork conditioning to perform many-to-many voice conversion between raw audio. Blow is trained end-to-end, with non-parallel data, on a frame-by-frame basis using a single speaker identifier. We show that Blow compares favorably to existing flow-based architectures and other competitive baselines, obtaining equal or better performance in both objective and subjective evaluations. We further assess the impact of its main components with an ablation study, and quantify a number of properties such as the necessary amount of training data or the preference for source or target speakers.

### Reference

J. Serrà, S. Pascual, & C. Segura (2019). **Blow: a single-scale hyperconditioned flow for non-parallel raw-audio voice conversion**. In _Advances in Neural Information Processing Systems (NeurIPS)_. In press.

```
@article{Serra19ARXIV,
author = {Serr{\`{a}}, J. and Pascual, S. and Segura, C.},
journal = {ArXiv},
title = {{Blow: a single-scale hyperconditioned flow for non-parallel raw-audio voice conversion}},
volume = {1906.00794},
year = {2019}
}
```

### Links

Paper: https://arxiv.org/abs/1906.00794 (latest version)

Audio examples: https://blowconversions.github.io

## Installation

Suggested steps are:

1. Clone repository.
1. Create a conda environment (you can use the `environment.yml` file).
1. The following folder structure will be produced by the repo. From the git folder:
    - `src/`: Where all scripts lie.
    - `dat/`: Place to put all preprocessed files (in subfolders).
    - `res/`: Place to save results.

## Running the code

All the following instructions assume you run them from the `src` folder. 
Also, check the arguments/code for the scripts below. You may want to run with a different configuration.

### Preprocessing

To preprocess the audio files:
```
python preprocess.py --path_in=/path/to/wav/root/folder/ --extension=.wav --path_out=../dat/pt/vctk
```
Our code expects audio filenames to be in the form `<speaker/class_id>_<utterance/track_id>_whatever.extension`, 
where elements inside `<>` do not contain the character `_` and IDs need not to be consecutive (example: `s001_u045_xxx.wav`). 
Therefore, if your data is not in this format, you should run or adapt the script `misc/rename_dataset.py`.

### Training

To train Blow:
```
python train.py --path_data=../dat/pt/vctk/ --path_out=../res/vctk/blow/ --model=blow
```

### Synthesis

To transform/synthesize audio with a given learnt model:
```
python synthesize.py --path_model=../res/vctk/blow/ --path_out=../res/vctk/blow/audio/ --convert
```

### Other

To execute the classification script:
```
python classify.py --mode=train --path_in=../dat/wav/vctk/train/ --fn_cla=../res/vctk/classif/trained_model.pt --fn_res=../res/vctk/classif/res_train.pt

python classify.py --mode=test --path_in=../res/vctk/blow/audio/ --fn_cla=../res/vctk/classif/trained_model.pt --fn_res=../res/vctk/classif/res_test.pt
```

To listen to some conversions (using sox's `play` command):
```
python misc/listening_test.py --path_refs_train=../dat/wav/vctk/train/ --path_refs_test=../dat/wav/vctk/test/ --paths_convs=../res/blow/audio/,../res/test1/audio/ --player=play
```

## Notes

- If using this code, parts of it, or developments from it, please cite the above reference.
- We do not provide any support or assistance for the supplied code nor we offer any other compilation/variant of it.
- We assume no responsibility regarding the provided code.

