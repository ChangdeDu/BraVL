import os
import argparse

import numpy as np
import torch
import scipy.io as sio
parser = argparse.ArgumentParser()

# TRAINING
parser.add_argument('--batch_size', type=int, default=1024, help="batch size for training")
parser.add_argument('--initial_learning_rate', type=float, default=0.0001, help="starting learning rate")
parser.add_argument('--beta_1', type=float, default=0.9, help="default beta_1 val for adam")
parser.add_argument('--beta_2', type=float, default=0.999, help="default beta_2 val for adam")
parser.add_argument('--start_epoch', type=int, default=0, help="flag to set the starting epoch for training")
parser.add_argument('--end_epoch', type=int, default=100, help="flag to indicate the final epoch of training")

# DATA DEPENDENT
parser.add_argument('--class_dim', type=int, default=32, help="dimension of common factor latent space")
# SAVE and LOAD
parser.add_argument('--mm_vae_save', type=str, default='mm_vae', help="model save for vae_bimodal")
parser.add_argument('--load_saved', type=bool, default=False, help="flag to indicate if a saved model will be loaded")

# DIRECTORIES
# experiments
parser.add_argument('--dir_experiment', type=str, default='./logs', help="directory to save logs in")
parser.add_argument('--dataname', type=str, default='ThingsEEG-Text', help="dataset")
parser.add_argument('--sbj', type=str, default='sub-01', help="eeg subject")
parser.add_argument('--roi', type=str, default='17channels', help="ROI")
parser.add_argument('--text_model', type=str, default='CLIPText', help="text embedding model")
parser.add_argument('--image_model', type=str, default='pytorch/cornet_s', help="image embedding model")

parser.add_argument('--test_type', type=str, default='zsl', help='normal or zsl')
parser.add_argument('--aug_type', type=str, default='no_aug', help='no_aug, image_text_ilsvrc2012_val')
parser.add_argument('--unimodal', type=str, default='image', help='image, text')
#multimodal
parser.add_argument('--method', type=str, default='joint_elbo', help='choose method for training the model')
parser.add_argument('--modality_jsd', type=bool, default=False, help="modality_jsd")
parser.add_argument('--modality_poe', type=bool, default=False, help="modality_poe")
parser.add_argument('--modality_moe', type=bool, default=False, help="modality_moe")
parser.add_argument('--joint_elbo', type=bool, default=False, help="modality_moe")
parser.add_argument('--poe_unimodal_elbos', type=bool, default=True, help="unimodal_klds")
parser.add_argument('--factorized_representation', action='store_true', default=False, help="factorized_representation")

# LOSS TERM WEIGHTS
parser.add_argument('--beta', type=float, default=0.0, help="default initial weight of sum of weighted divergence terms")
parser.add_argument('--beta_style', type=float, default=1.0, help="default weight of sum of weighted style divergence terms")
parser.add_argument('--beta_content', type=float, default=1.0, help="default weight of sum of weighted content divergence terms")
parser.add_argument('--lambda1', type=float, default=0.001, help="default weight of intra_mi terms")
parser.add_argument('--lambda2', type=float, default=0.001, help="default weight of inter_mi terms")


FLAGS = parser.parse_args()
data_dir_root = os.path.join('./data', FLAGS.dataname)
brain_dir = os.path.join(data_dir_root, 'brain_feature', FLAGS.roi, FLAGS.sbj)
image_dir_train = os.path.join(data_dir_root, 'visual_feature/ThingsTrain', FLAGS.image_model, FLAGS.sbj)
text_dir_train = os.path.join(data_dir_root, 'textual_feature/ThingsTrain/text', FLAGS.text_model, FLAGS.sbj)

train_brain = sio.loadmat(os.path.join(brain_dir, 'eeg_train_data_within.mat'))['data'].astype('double')
train_brain = train_brain[:,:,27:60] # 70ms-400ms
train_brain = np.reshape(train_brain,(train_brain.shape[0],-1))
train_image = sio.loadmat(os.path.join(image_dir_train, 'feat_pca_train.mat'))['data'].astype('double')
train_text = sio.loadmat(os.path.join(text_dir_train, 'text_feat_train.mat'))['data'].astype('double')
train_image = train_image[:,0:100] # top 100 PCs


train_brain = torch.from_numpy(train_brain)
train_image = torch.from_numpy(train_image)
train_text = torch.from_numpy(train_text)
dim_brain = train_brain.shape[1]
dim_image = train_image.shape[1]
dim_text = train_text.shape[1]

parser.add_argument('--m1_dim', type=int, default=dim_brain, help="dimension of modality brain")
parser.add_argument('--m2_dim', type=int, default=dim_image, help="dimension of modality image")
parser.add_argument('--m3_dim', type=int, default=dim_text, help="dimension of modality text")
parser.add_argument('--data_dir_root', type=str, default=data_dir_root, help="data dir")

FLAGS = parser.parse_args()
print(FLAGS)
