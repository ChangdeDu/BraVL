# BraVL
This is the official code for the paper "Beyond the Eye: Decoding Visual Neural Representations by Multimodal Learning of Brain-Visual-Linguistic Features".

## Preliminaries

This code was developed and tested with:
- Python version 3.7.0
- PyTorch version 1.9.0
- CUDA version 11.2
- The conda environment defined in `environment.yml`

First, set up the conda enviroment as follows:
```bash
conda env create -f environment.yml  # create conda env
conda activate BraVL                # activate conda env
```
## Download data
Second, download the pre-processed trimodal data from https://figshare.com/articles/dataset/BraVL/17024591, unzip them, and put them at "./data" directory:
```bash
unzip DIR-Wiki.zip -d data/
unzip GOD-Wiki.zip -d data/
```
Note that, the raw (image and brain fMRI) data are not included here because they are too large. Raw ImageNet images and brain fMRI data can be downloaded from the corresponding official site. We provide python scripts for feature extraction and data preprocessing.

## Experiments

Experiments can be started by running the `job_trimodal` script.


### running BraVL
```bash
bash job_trimodal
```

If you have any questions about the code or the paper, we are happy to help!
