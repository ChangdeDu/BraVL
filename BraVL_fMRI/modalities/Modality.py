
from abc import ABC, abstractmethod
import os

import torch
import torch.distributions as dist

class Modality(ABC):
    def __init__(self, name, enc, dec, class_dim, style_dim, lhood_name):
        self.name = name;
        self.encoder = enc;
        self.decoder = dec;
        self.class_dim = class_dim;
        self.style_dim = style_dim;
        self.likelihood_name = lhood_name;
        self.likelihood = self.get_likelihood(lhood_name);


    def get_likelihood(self, name):
        if name == 'laplace':
            pz = dist.Laplace;
        elif name == 'bernoulli':
            pz = dist.Bernoulli;
        elif name == 'normal':
            pz = dist.Normal;
        elif name == 'categorical':
            pz = dist.OneHotCategorical;
        else:
            print('likelihood not implemented')
            pz = None;
        return pz;





    def calc_log_prob(self, out_dist, target, norm_value):
        log_prob = out_dist.log_prob(target).sum();
        mean_val_logprob = log_prob/norm_value;
        return mean_val_logprob;


    def save_networks(self, dir_checkpoints):
        torch.save(self.encoder.state_dict(), os.path.join(dir_checkpoints,
                                                           'enc_' + self.name))
        torch.save(self.decoder.state_dict(), os.path.join(dir_checkpoints,
                                                           'dec_' + self.name))

