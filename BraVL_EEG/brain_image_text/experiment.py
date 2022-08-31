import os
import numpy as np 
import itertools
import scipy.io as sio
import torch
import torch.optim as optim
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset
from modalities.Modality import Modality
from brain_image_text.networks.VAEtrimodal import VAEtrimodal,VAEbimodal
from brain_image_text.networks.QNET import QNet
from brain_image_text.networks.MLP_Brain import EncoderBrain, DecoderBrain
from brain_image_text.networks.MLP_Image import EncoderImage, DecoderImage
from brain_image_text.networks.MLP_Text import EncoderText, DecoderText
from utils.BaseExperiment import BaseExperiment


class BrainImageText(BaseExperiment):
    def __init__(self, flags, alphabet):
        super().__init__(flags)

        self.modalities = self.set_modalities()
        self.num_modalities = len(self.modalities.keys())
        self.subsets = self.set_subsets()
        self.dataset_train = None
        self.dataset_test = None

        self.set_dataset()
        self.mm_vae = self.set_model()
        self.optimizer = None
        self.rec_weights = self.set_rec_weights()
        self.style_weights = self.set_style_weights()
        self.Q1,self.Q2,self.Q3 = self.set_Qmodel()
        self.eval_metric = accuracy_score

        self.labels = ['digit']


    def set_model(self):
        model = VAEtrimodal(self.flags, self.modalities, self.subsets)
        model = model.to(self.flags.device)
        return model

    def set_modalities(self):
        mod1 = Modality('brain', EncoderBrain(self.flags), DecoderBrain(self.flags),
                    self.flags.class_dim, self.flags.style_m1_dim, 'normal')
        mod2 = Modality('image', EncoderImage(self.flags), DecoderImage(self.flags),
                    self.flags.class_dim, self.flags.style_m2_dim, 'normal')
        mod3 = Modality('text', EncoderText(self.flags), DecoderText(self.flags),
                    self.flags.class_dim, self.flags.style_m3_dim, 'normal')
        mods = {mod1.name: mod1, mod2.name: mod2, mod3.name: mod3}
        return mods

    def set_dataset(self):
        # load data
        data_dir_root = self.flags.data_dir_root
        sbj = self.flags.sbj
        image_model = self.flags.image_model
        text_model = self.flags.text_model
        roi = self.flags.roi
        brain_dir = os.path.join(data_dir_root, 'brain_feature', roi, sbj)
        image_dir_train = os.path.join(data_dir_root, 'visual_feature/ThingsTrain', image_model, sbj)
        image_dir_test = os.path.join(data_dir_root, 'visual_feature/ThingsTest', image_model, sbj)
        text_dir_train = os.path.join(data_dir_root, 'textual_feature/ThingsTrain/text', text_model, sbj)
        text_dir_test = os.path.join(data_dir_root, 'textual_feature/ThingsTest/text', text_model, sbj)

        train_brain = sio.loadmat(os.path.join(brain_dir, 'eeg_train_data_within.mat'))['data'].astype('double') * 2.0
        # train_brain = sio.loadmat(os.path.join(brain_dir, 'eeg_train_data_between.mat'))['data'].astype('double')*2.0
        train_brain = train_brain[:,:,27:60] # 70ms-400ms
        train_brain = np.reshape(train_brain, (train_brain.shape[0], -1))
        train_image = sio.loadmat(os.path.join(image_dir_train, 'feat_pca_train.mat'))['data'].astype('double')*50.0
        train_text = sio.loadmat(os.path.join(text_dir_train, 'text_feat_train.mat'))['data'].astype('double')*2.0
        train_label = sio.loadmat(os.path.join(brain_dir, 'eeg_train_data_within.mat'))['class_idx'].T.astype('int')
        train_image = train_image[:,0:100]

        # test_brain = sio.loadmat(os.path.join(brain_dir, 'eeg_test_data_unique.mat'))['data'].astype('double')*2.0
        # test_brain = test_brain[:, :, 27:60]
        # test_brain = np.reshape(test_brain, (test_brain.shape[0], -1))
        # test_image = sio.loadmat(os.path.join(image_dir_test, 'feat_pca_test_unique.mat'))['data'].astype('double')*50.0
        # test_text = sio.loadmat(os.path.join(text_dir_test, 'text_feat_test_unique.mat'))['data'].astype('double')*2.0
        # test_label = sio.loadmat(os.path.join(brain_dir, 'eeg_test_data_unique.mat'))['class_idx'].T.astype('int')
        # train_image = train_image[:, 0:100]


        test_brain = sio.loadmat(os.path.join(brain_dir, 'eeg_test_data.mat'))['data'].astype('double')*2.0
        test_brain = test_brain[:, :, 27:60]
        test_brain = np.reshape(test_brain, (test_brain.shape[0], -1))
        test_image = sio.loadmat(os.path.join(image_dir_test, 'feat_pca_test.mat'))['data'].astype('double')*50.0
        test_text = sio.loadmat(os.path.join(text_dir_test, 'text_feat_test.mat'))['data'].astype('double')*2.0
        test_label = sio.loadmat(os.path.join(brain_dir, 'eeg_test_data.mat'))['class_idx'].T.astype('int')
        test_image = test_image[:, 0:100]

        if self.flags.aug_type == 'image_text_ilsvrc2012_val':
            image_dir_aug = os.path.join(data_dir_root, 'visual_feature/Aug_ILSVRC2012_val', image_model, sbj)
            text_dir_aug = os.path.join(data_dir_root, 'textual_feature/Aug_ILSVRC2012_val/text', text_model, sbj)
            aug_image = sio.loadmat(os.path.join(image_dir_aug, 'feat_pca_aug_ilsvrc2012_val.mat'))['data'].astype('double')
            aug_image = aug_image[:, 0:100]
            aug_text = sio.loadmat(os.path.join(text_dir_aug, 'text_feat_aug_ilsvrc2012_val.mat'))['data'].astype('double')
            aug_image = torch.from_numpy(aug_image)
            aug_text = torch.from_numpy(aug_text)
            print('aug_image=', aug_image.shape)
            print('aug_text=', aug_text.shape)
        elif self.flags.aug_type == 'no_aug':
            print('no augmentation')

        if self.flags.test_type=='normal':
            train_label_stratify = train_label
            train_brain, val_brain, train_label, val_label = train_test_split(train_brain, train_label_stratify, test_size=0.2, stratify=train_label_stratify)
            train_image, val_image, train_label, val_label = train_test_split(train_image, train_label_stratify, test_size=0.2, stratify=train_label_stratify)
            train_text, val_text, train_label, val_label = train_test_split(train_text, train_label_stratify, test_size=0.2, stratify=train_label_stratify)

            val_brain = torch.from_numpy(val_brain)
            val_image = torch.from_numpy(val_image)
            val_text = torch.from_numpy(val_text)
            val_label = torch.from_numpy(val_label)
            print('val_brain=', val_brain.shape)
            print('val_image=', val_image.shape)
            print('val_text=', val_text.shape)

        train_brain = torch.from_numpy(train_brain)
        test_brain = torch.from_numpy(test_brain)
        train_image = torch.from_numpy(train_image)
        test_image = torch.from_numpy(test_image)
        train_text = torch.from_numpy(train_text)
        test_text = torch.from_numpy(test_text)
        train_label = torch.from_numpy(train_label)
        test_label = torch.from_numpy(test_label)


        print('train_brain=', train_brain.shape)
        print('train_image=', train_image.shape)
        print('train_text=', train_text.shape)
        print('test_brain=', test_brain.shape)
        print('test_image=', test_image.shape)
        print('test_text=', test_text.shape)

        self.m1_dim = train_brain.shape[1]
        self.m2_dim = train_image.shape[1]
        self.m3_dim = train_text.shape[1]

        train_dataset = torch.utils.data.TensorDataset(train_brain, train_image, train_text, train_label)
        test_dataset = torch.utils.data.TensorDataset(test_brain, test_image, test_text,test_label)

        self.dataset_train = train_dataset
        self.dataset_test = test_dataset

        if self.flags.test_type == 'normal':
            val_dataset = torch.utils.data.TensorDataset(val_brain, val_image, val_text, val_label)
            self.dataset_val = val_dataset

        if 'image_text' in self.flags.aug_type:
            aug_dataset = torch.utils.data.TensorDataset(aug_image, aug_text)
            self.dataset_aug = aug_dataset
        elif self.flags.aug_type == 'no_aug':
            print('no augmentation')


    def set_optimizer(self):
        optimizer = optim.Adam(
            itertools.chain(self.mm_vae.parameters(),self.Q1.parameters(),self.Q2.parameters(),self.Q3.parameters()),
            lr=self.flags.initial_learning_rate,
            betas=(self.flags.beta_1, self.flags.beta_2))
        optimizer_mvae = optim.Adam(
            list(self.mm_vae.parameters()),
            lr=self.flags.initial_learning_rate,
            betas=(self.flags.beta_1, self.flags.beta_2))
        optimizer_Qnet = optim.Adam(
            itertools.chain(self.Q1.parameters(),self.Q2.parameters(),self.Q3.parameters()),
            lr=self.flags.initial_learning_rate,
            betas=(self.flags.beta_1, self.flags.beta_2))
        self.optimizer = {'mvae':optimizer_mvae,'Qnet':optimizer_Qnet,'all':optimizer}
        scheduler_mvae = optim.lr_scheduler.StepLR(optimizer_mvae, step_size=20, gamma=1.0)
        scheduler_Qnet = optim.lr_scheduler.StepLR(optimizer_Qnet, step_size=20, gamma=1.0)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=1.0)
        self.scheduler = {'mvae': scheduler_mvae, 'Qnet': scheduler_Qnet, 'all': scheduler}


    def set_Qmodel(self):
        Q1 = QNet(input_dim=self.flags.m1_dim, latent_dim=self.flags.class_dim).cuda()
        Q2 = QNet(input_dim=self.flags.m2_dim, latent_dim=self.flags.class_dim).cuda()
        Q3 = QNet(input_dim=self.flags.m3_dim, latent_dim=self.flags.class_dim).cuda()
        return Q1, Q2 ,Q3

    def set_rec_weights(self):
        weights = dict()
        weights['brain'] = self.flags.beta_m1_rec
        weights['image'] = self.flags.beta_m2_rec
        weights['text'] = self.flags.beta_m3_rec
        return weights

    def set_style_weights(self):
        weights = dict()
        weights['brain'] = self.flags.beta_m1_style
        weights['image'] = self.flags.beta_m2_style
        weights['text'] = self.flags.beta_m3_style
        return weights
