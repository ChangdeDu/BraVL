
import torch
import torch.nn as nn

from divergence_measures.kl_div import calc_kl_divergence
from divergence_measures.kl_div import calc_kl_divergence_lb_gauss_mixture
from divergence_measures.kl_div import calc_kl_divergence_ub_gauss_mixture
from divergence_measures.kl_div import calc_entropy_gauss

from utils.utils import reweight_weights


def poe(mu, logvar, eps=1e-8):
    var = torch.exp(logvar) + eps
    # precision of i-th Gaussian expert at point x
    T = 1. / var
    pd_mu = torch.sum(mu * T, dim=0) / torch.sum(T, dim=0)
    pd_var = 1. / torch.sum(T, dim=0)
    pd_logvar = torch.log(pd_var)
    return pd_mu, pd_logvar


def alpha_poe(alpha, mu, logvar, eps=1e-8):
    var = torch.exp(logvar) + eps
    # precision of i-th Gaussian expert at point x
    if var.dim() == 3:
        alpha_expanded = alpha.unsqueeze(-1).unsqueeze(-1);
    elif var.dim() == 4:
        alpha_expanded = alpha.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1);

    T = 1 / var;
    pd_var = 1. / torch.sum(alpha_expanded * T, dim=0)
    pd_mu = pd_var * torch.sum(alpha_expanded * mu * T, dim=0)
    pd_logvar = torch.log(pd_var)
    return pd_mu, pd_logvar;


def calc_alphaJSD_modalities_mixture(m1_mu, m1_logvar, m2_mu, m2_logvar, flags):
    klds = torch.zeros(2);
    entropies_mixture = torch.zeros(2);
    w_modalities = torch.Tensor(flags.alpha_modalities[1:]);
    if flags.cuda:
        w_modalities = w_modalities.cuda();
        klds = klds.cuda();
        entropies_mixture = entropies_mixture.cuda();
    w_modalities = reweight_weights(w_modalities);

    mus = [m1_mu, m2_mu]
    logvars = [m1_logvar, m2_logvar]
    for k in range(0, len(mus)):
        ent = calc_entropy_gauss(flags, logvars[k], norm_value=flags.batch_size);
        # print('entropy: ' + str(ent))
        # print('lb: ' )
        kld_lb = calc_kl_divergence_lb_gauss_mixture(flags, k, mus[k], logvars[k], mus, logvars,
                                                     norm_value=flags.batch_size);
        print('kld_lb: ' + str(kld_lb))
        # print('ub: ')
        kld_ub = calc_kl_divergence_ub_gauss_mixture(flags, k, mus[k], logvars[k], mus, logvars, ent,
                                                     norm_value=flags.batch_size);
        print('kld_ub: ' + str(kld_ub))
        # kld_mean = (kld_lb+kld_ub)/2;
        entropies_mixture[k] = ent.clone();
        klds[k] = 0.5*(kld_lb + kld_ub);
        # klds[k] = kld_ub;
    summed_klds = (w_modalities * klds).sum();
    # print('summed klds: ' + str(summed_klds));
    return summed_klds, klds, entropies_mixture;

def calc_alphaJSD_modalities(flags, mus, logvars, weights, normalization=None):
    num_mods = mus.shape[0];
    num_samples = mus.shape[1];
    alpha_mu, alpha_logvar = alpha_poe(weights, mus, logvars)
    if normalization is not None:
        klds = torch.zeros(num_mods);
    else:
        klds = torch.zeros(num_mods, num_samples);
    klds = klds.to(flags.device);

    for k in range(0, num_mods):
        kld = calc_kl_divergence(mus[k,:,:], logvars[k,:,:], alpha_mu,
                                 alpha_logvar, norm_value=normalization);
        if normalization is not None:
            klds[k] = kld;
        else:
            klds[k,:] = kld;
    if normalization is None:
        weights = weights.unsqueeze(1).repeat(1, num_samples);
    group_div = (weights * klds).sum(dim=0);
    return group_div, klds, [alpha_mu, alpha_logvar];


def calc_group_divergence_moe(flags, mus, logvars, weights, normalization=None):
    num_mods = mus.shape[0];
    num_samples = mus.shape[1];
    if normalization is not None:
        klds = torch.zeros(num_mods);
    else:
        klds = torch.zeros(num_mods, num_samples);
    klds = klds.to(flags.device);
    weights = weights.to(flags.device);
    for k in range(0, num_mods):
        kld_ind = calc_kl_divergence(mus[k,:,:], logvars[k,:,:],
                                     norm_value=normalization);
        if normalization is not None:
            klds[k] = kld_ind;
        else:
            klds[k,:] = kld_ind;
    if normalization is None:
        weights = weights.unsqueeze(1).repeat(1, num_samples);
    group_div = (weights*klds).sum(dim=0);
    return group_div, klds;


def calc_group_divergence_poe(flags, mus, logvars, norm=None):
    num_mods = mus.shape[0];
    poe_mu, poe_logvar = poe(mus, logvars)
    kld_poe = calc_kl_divergence(poe_mu, poe_logvar, norm_value=norm);
    klds = torch.zeros(num_mods).to(flags.device);
    for k in range(0, num_mods):
        kld_ind = calc_kl_divergence(mus[k,:,:], logvars[k,:,:],
                                     norm_value=norm);
        klds[k] = kld_ind;
    return kld_poe, klds, [poe_mu, poe_logvar];


def calc_modality_divergence(m1_mu, m1_logvar, m2_mu, m2_logvar, flags):
    if flags.modality_poe:
        kld_batch = calc_kl_divergence(m1_mu, m1_logvar, m2_mu, m2_logvar, norm_value=flags.batch_size).sum();
        return kld_batch;
    else:
        uniform_mu = torch.zeros(m1_mu.shape)
        uniform_logvar = torch.zeros(m1_logvar.shape)
        klds = torch.zeros(3,3)
        klds_modonly = torch.zeros(2,2)
        if flags.cuda:
            klds = klds.cuda();
            klds_modonly = klds_modonly.cuda();
            uniform_mu = uniform_mu.cuda();
            uniform_logvar = uniform_logvar.cuda();

        mus = [uniform_mu, m1_mu, m2_mu]
        logvars = [uniform_logvar, m1_logvar, m2_logvar]
        for i in range(1, len(mus)): # CAREFUL: index starts from one, not zero
            for j in range(0, len(mus)):
                kld = calc_kl_divergence(mus[i], logvars[i], mus[j], logvars[j], norm_value=flags.batch_size);
                klds[i,j] = kld;
                if i >= 1 and j >= 1:
                    klds_modonly[i-1,j-1] = kld;
        klds = klds.sum()/(len(mus)*(len(mus)-1))
        klds_modonly = klds_modonly.sum()/((len(mus)-1)*(len(mus)-1));
        return [klds, klds_modonly];
