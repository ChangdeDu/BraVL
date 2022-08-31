from abc import ABC, abstractmethod

import os

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.distributions as dist
from divergence_measures.mm_div import calc_alphaJSD_modalities
from divergence_measures.mm_div import calc_group_divergence_moe
from divergence_measures.mm_div import poe

from utils import utils


class BaseMMVae(ABC, nn.Module):
    def __init__(self, flags, modalities, subsets):
        super(BaseMMVae, self).__init__()
        self.num_modalities = len(modalities.keys());
        self.flags = flags;
        self.modalities = modalities;
        self.subsets = subsets;
        self.set_fusion_functions();

        encoders = nn.ModuleDict();
        decoders = nn.ModuleDict();
        lhoods = dict();
        for m, m_key in enumerate(sorted(modalities.keys())):
            encoders[m_key] = modalities[m_key].encoder;
            decoders[m_key] = modalities[m_key].decoder;
            lhoods[m_key] = modalities[m_key].likelihood;
        self.encoders = encoders;
        self.decoders = decoders;
        self.lhoods = lhoods;


    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = Variable(std.data.new(std.size()).normal_())
        return eps.mul(std).add_(mu)


    def set_fusion_functions(self):
        weights = utils.reweight_weights(torch.Tensor(self.flags.alpha_modalities));
        self.weights = weights.to(self.flags.device);
        if self.flags.modality_moe:
            self.modality_fusion = self.moe_fusion;
            self.fusion_condition = self.fusion_condition_moe; self.calc_joint_divergence = self.divergence_static_prior;
        elif self.flags.modality_jsd:
            self.modality_fusion = self.moe_fusion;
            self.fusion_condition = self.fusion_condition_moe;
            self.calc_joint_divergence = self.divergence_dynamic_prior;
        elif self.flags.modality_poe:
            self.modality_fusion = self.poe_fusion;
            self.fusion_condition = self.fusion_condition_poe;
            self.calc_joint_divergence = self.divergence_static_prior;
        elif self.flags.joint_elbo:
            self.modality_fusion = self.poe_fusion;
            self.fusion_condition = self.fusion_condition_joint;
            self.calc_joint_divergence = self.divergence_static_prior;


    def divergence_static_prior(self, mus, logvars, weights=None):
        if weights is None:
            weights=self.weights;
        weights = weights.clone();
        weights = utils.reweight_weights(weights);
        div_measures = calc_group_divergence_moe(self.flags,
                                                 mus,
                                                 logvars,
                                                 weights,
                                                 normalization=self.flags.batch_size);
        divs = dict();
        divs['joint_divergence'] = div_measures[0]; divs['individual_divs'] = div_measures[1]; divs['dyn_prior'] = None;
        return divs;


    def divergence_dynamic_prior(self, mus, logvars, weights=None):
        if weights is None:
            weights = self.weights;
        div_measures = calc_alphaJSD_modalities(self.flags,
                                                mus,
                                                logvars,
                                                weights,
                                                normalization=self.flags.batch_size);
        divs = dict();
        divs['joint_divergence'] = div_measures[0];
        divs['individual_divs'] = div_measures[1];
        divs['dyn_prior'] = div_measures[2];
        return divs;


    def moe_fusion(self, mus, logvars, weights=None):
        if weights is None:
            weights = self.weights;
        weights = utils.reweight_weights(weights);
        #mus = torch.cat(mus, dim=0);
        #logvars = torch.cat(logvars, dim=0);
        mu_moe, logvar_moe = utils.mixture_component_selection(self.flags,
                                                               mus,
                                                               logvars,
                                                               weights);
        return [mu_moe, logvar_moe];


    def poe_fusion(self, mus, logvars, weights=None):
        if (self.flags.modality_poe or mus.shape[0] == 
            len(self.modalities.keys())):
            num_samples = mus[0].shape[0];
            mus = torch.cat((mus, torch.zeros(1, num_samples,
                             self.flags.class_dim).to(self.flags.device)),
                            dim=0);
            logvars = torch.cat((logvars, torch.zeros(1, num_samples,
                                 self.flags.class_dim).to(self.flags.device)),
                                dim=0);
        #mus = torch.cat(mus, dim=0);
        #logvars = torch.cat(logvars, dim=0);
        mu_poe, logvar_poe = poe(mus, logvars);
        return [mu_poe, logvar_poe];


    def fusion_condition_moe(self, subset, input_batch=None):
        if len(subset) == 1:
            return True;
        else:
            return False;


    def fusion_condition_poe(self, subset, input_batch=None):
        if len(subset) == len(input_batch.keys()):
            return True;
        else:
            return False;


    def fusion_condition_joint(self, subset, input_batch=None):
        return True;


    def forward(self, input_batch,K=1):
        latents = self.inference(input_batch);
        results = dict();
        results['latents'] = latents;
        results['group_distr'] = latents['joint'];
        class_embeddings = self.reparameterize(latents['joint'][0],
                                                latents['joint'][1]);
        #### For CUBO ####
        qz_x = dist.Normal(latents['joint'][0],latents['joint'][1].mul(0.5).exp_())
        zss = qz_x.rsample(torch.Size([K]))

        div = self.calc_joint_divergence(latents['mus'],
                                         latents['logvars'],
                                         latents['weights']);
        for k, key in enumerate(div.keys()):
            results[key] = div[key];

        results_rec = dict();
        px_zs = dict();
        enc_mods = latents['modalities'];
        for m, m_key in enumerate(self.modalities.keys()):
            if m_key in input_batch.keys():
                m_s_mu, m_s_logvar = enc_mods[m_key + '_style'];
                if self.flags.factorized_representation:
                    m_s_embeddings = self.reparameterize(mu=m_s_mu, logvar=m_s_logvar);
                else:
                    m_s_embeddings = None;
                m_rec = self.lhoods[m_key](*self.decoders[m_key](m_s_embeddings, class_embeddings));
                px_z = self.lhoods[m_key](*self.decoders[m_key](m_s_embeddings, zss));
                results_rec[m_key] = m_rec;
                px_zs[m_key] = px_z
        results['rec'] = results_rec;
        results['class_embeddings'] = class_embeddings
        results['qz_x'] = qz_x
        results['zss'] = zss
        results['px_zs'] = px_zs
        return results;

    def encode(self, input_batch):
        latents = dict();
        for m, m_key in enumerate(self.modalities.keys()):
            if m_key in input_batch.keys():
                i_m = input_batch[m_key];
                l = self.encoders[m_key](i_m)
                latents[m_key + '_style'] = l[:2]
                latents[m_key] = l[2:]
            else:
                latents[m_key + '_style'] = [None, None];
                latents[m_key] = [None, None];
        return latents;


    def inference(self, input_batch, num_samples=None):
        if num_samples is None:
            num_samples = self.flags.batch_size;
        latents = dict();
        enc_mods = self.encode(input_batch);
        latents['modalities'] = enc_mods;
        mus = torch.Tensor().to(self.flags.device);
        logvars = torch.Tensor().to(self.flags.device);
        distr_subsets = dict();
        for k, s_key in enumerate(self.subsets.keys()):
            if s_key != '':
                mods = self.subsets[s_key];
                mus_subset = torch.Tensor().to(self.flags.device);
                logvars_subset = torch.Tensor().to(self.flags.device);
                mods_avail = True
                for m, mod in enumerate(mods):
                    if mod.name in input_batch.keys():
                        mus_subset = torch.cat((mus_subset,
                                                enc_mods[mod.name][0].unsqueeze(0)),
                                               dim=0);
                        logvars_subset = torch.cat((logvars_subset,
                                                    enc_mods[mod.name][1].unsqueeze(0)),
                                                   dim=0);
                    else:
                        mods_avail = False;
                if mods_avail:
                    weights_subset = ((1/float(len(mus_subset)))*
                                      torch.ones(len(mus_subset)).to(self.flags.device));
                    s_mu, s_logvar = self.modality_fusion(mus_subset,
                                                          logvars_subset,
                                                          weights_subset); #子集内部POE#
                    distr_subsets[s_key] = [s_mu, s_logvar];
                    if self.fusion_condition(mods, input_batch):
                        mus = torch.cat((mus, s_mu.unsqueeze(0)), dim=0);
                        logvars = torch.cat((logvars, s_logvar.unsqueeze(0)),
                                            dim=0);
        if self.flags.modality_jsd:
            num_samples = mus[0].shape[0]
            mus = torch.cat((mus, torch.zeros(1, num_samples,
                                      self.flags.class_dim).to(self.flags.device)),
                            dim=0);
            logvars = torch.cat((logvars, torch.zeros(1, num_samples,
                                          self.flags.class_dim).to(self.flags.device)),
                                dim=0);
        #weights = (1/float(len(mus)))*torch.ones(len(mus)).to(self.flags.device);
        weights = (1/float(mus.shape[0]))*torch.ones(mus.shape[0]).to(self.flags.device);
        joint_mu, joint_logvar = self.moe_fusion(mus, logvars, weights); #子集之间MOE#
        #mus = torch.cat(mus, dim=0);
        #logvars = torch.cat(logvars, dim=0);
        latents['mus'] = mus;
        latents['logvars'] = logvars;
        latents['weights'] = weights;
        latents['joint'] = [joint_mu, joint_logvar];
        latents['subsets'] = distr_subsets;
        return latents;


    def generate(self, num_samples=None):
        if num_samples is None:
            num_samples = self.flags.batch_size;

        mu = torch.zeros(num_samples,
                         self.flags.class_dim).to(self.flags.device);
        logvar = torch.zeros(num_samples,
                             self.flags.class_dim).to(self.flags.device);
        z_class = self.reparameterize(mu, logvar);
        z_styles = self.get_random_styles(num_samples);
        random_latents = {'content': z_class, 'style': z_styles};
        random_samples = self.generate_from_latents(random_latents);
        return random_samples;


    def generate_sufficient_statistics_from_latents(self, latents):
        suff_stats = dict();
        content = latents['content']
        for m, m_key in enumerate(self.modalities.keys()):
            s = latents['style'][m_key];
            cg = self.lhoods[m_key](*self.decoders[m_key](s, content));
            suff_stats[m_key] = cg;
        return suff_stats;


    def generate_from_latents(self, latents):
        suff_stats = self.generate_sufficient_statistics_from_latents(latents);
        cond_gen = dict();
        for m, m_key in enumerate(latents['style'].keys()):
            cond_gen_m = suff_stats[m_key].mean;
            cond_gen[m_key] = cond_gen_m;
        return cond_gen;


    def cond_generation(self, latent_distributions, num_samples=None):
        if num_samples is None:
            num_samples = self.flags.batch_size;

        style_latents = self.get_random_styles(num_samples);
        cond_gen_samples = dict();
        for k, key in enumerate(latent_distributions.keys()):
            [mu, logvar] = latent_distributions[key];
            content_rep = self.reparameterize(mu=mu, logvar=logvar);
            latents = {'content': content_rep, 'style': style_latents}
            cond_gen_samples[key] = self.generate_from_latents(latents);
        return cond_gen_samples;


    def get_random_style_dists(self, num_samples):
        styles = dict();
        for k, m_key in enumerate(self.modalities.keys()):
            mod = self.modalities[m_key];
            s_mu = torch.zeros(num_samples,
                               mod.style_dim).to(self.flags.device)
            s_logvar = torch.zeros(num_samples,
                                   mod.style_dim).to(self.flags.device);
            styles[m_key] = [s_mu, s_logvar];
        return styles;


    def get_random_styles(self, num_samples):
        styles = dict();
        for k, m_key in enumerate(self.modalities.keys()):
            if self.flags.factorized_representation:
                mod = self.modalities[m_key];
                z_style = torch.randn(num_samples, mod.style_dim);
                z_style = z_style.to(self.flags.device);
            else:
                z_style = None;
            styles[m_key] = z_style;
        return styles;


    def save_networks(self):
        for k, m_key in enumerate(self.modalities.keys()):
            torch.save(self.encoders[m_key].state_dict(),
                       os.path.join(self.flags.dir_checkpoints, 'enc_' +
                                    self.modalities[m_key].name))
            torch.save(self.decoders[m_key].state_dict(),
                       os.path.join(self.flags.dir_checkpoints, 'dec_' +
                                    self.modalities[m_key].name))




