import math
import torch

from utils.utils import reweight_weights


def calc_kl_divergence(mu0, logvar0, mu1=None, logvar1=None, norm_value=None):
    if mu1 is None or logvar1 is None:
        KLD = -0.5 * torch.sum(1 - logvar0.exp() - mu0.pow(2) + logvar0)
    else:
        KLD = -0.5 * (torch.sum(1 - logvar0.exp()/logvar1.exp() - (mu0-mu1).pow(2)/logvar1.exp() + logvar0 - logvar1))
    if norm_value is not None:
        KLD = KLD / float(norm_value);
    return KLD


def calc_gaussian_scaling_factor(PI, mu1, logvar1, mu2=None, logvar2=None, norm_value=None):
    d = mu1.shape[1];
    if mu2 is None or logvar2 is None:
        # print('S_11: ' + str(torch.sum(1/((2*PI*(logvar1.exp() + 1)).pow(0.5)))))
        # print('S_12: ' + str(torch.sum(torch.exp(-0.5*(mu1.pow(2)/(logvar1.exp()+1))))))
        S_pre = (1/(2*PI).pow(d/2))*torch.sum((logvar1.exp() + 1), dim=1).pow(0.5);
        S = S_pre*torch.sum((-0.5*(mu1.pow(2)/(logvar1.exp()+1))).exp(), dim=1);
        S = torch.sum(S)
    else:
        # print('S_21: ' + str(torch.sum(1/((2*PI).pow(d/2)*(logvar1.exp()+logvar2.exp()).pow(0.5)))));
        # print('S_22: ' + str(torch.sum(torch.exp(-0.5 * ((mu1 - mu2).pow(2) / (logvar1.exp() + logvar2.exp()))))));
        S_pre = torch.sum(1/((2*PI).pow(d/2)*(logvar1.exp()+logvar2.exp())), dim=1).pow(0.5)
        S = S_pre*torch.sum(torch.exp(-0.5*((mu1-mu2).pow(2)/(logvar1.exp()+logvar2.exp()))), dim=1);
        S = torch.sum(S)
    if norm_value is not None:
        S = S / float(norm_value);
    # print('S: ' + str(S))
    return S


def calc_gaussian_scaling_factor_self(PI, logvar1, norm_value=None):
    d = logvar1.shape[1];
    S = (1/(2*PI).pow(d/2))*torch.sum(logvar1.exp(), dim=1).pow(0.5);
    S = torch.sum(S);
    # S = torch.sum(1 / (2*(PI*torch.exp(logvar1)).pow(0.5)));
    if norm_value is not None:
        S = S / float(norm_value);
    # print('S self: ' + str(S))
    return S


#def calc_kl_divergence_lb_gauss_mixture(flags, index, mu1, logvar1, mus, logvars, norm_value=None):
#     klds = torch.zeros(mus.shape[0]+1)
#     if flags.cuda:
#         klds = klds.cuda();
#
#     klds[0] = calc_kl_divergence(mu1, logvar1, norm_value=norm_value);
#     for k in range(0, mus.shape[0]):
#         if k == index:
#             kld = 0.0;
#         else:
#             kld = calc_kl_divergence(mu1, logvar1, mus[k], logvars[k], norm_value=norm_value);
#         klds[k+1] = kld;
#     kld_mixture = klds.mean();
#     return kld_mixture;

def calc_kl_divergence_lb_gauss_mixture(flags, index, mu1, logvar1, mus, logvars, norm_value=None):
    PI = torch.Tensor([math.pi]);
    w_modalities = torch.Tensor(flags.alpha_modalities);
    if flags.cuda:
        PI = PI.cuda();
        w_modalities = w_modalities.cuda();
    w_modalities = reweight_weights(w_modalities);

    denom = w_modalities[0]*calc_gaussian_scaling_factor(PI, mu1, logvar1, norm_value=norm_value);
    for k in range(0, len(mus)):
        if index == k:
            denom += w_modalities[k+1]*calc_gaussian_scaling_factor_self(PI, logvar1, norm_value=norm_value);
        else:
            denom += w_modalities[k+1]*calc_gaussian_scaling_factor(PI, mu1, logvar1, mus[k], logvars[k], norm_value=norm_value)
    lb = -torch.log(denom);
    return lb;


def calc_kl_divergence_ub_gauss_mixture(flags, index, mu1, logvar1, mus, logvars, entropy, norm_value=None):
    PI = torch.Tensor([math.pi]);
    w_modalities = torch.Tensor(flags.alpha_modalities);
    if flags.cuda:
        PI = PI.cuda();
        w_modalities = w_modalities.cuda();
    w_modalities = reweight_weights(w_modalities);

    nom = calc_gaussian_scaling_factor_self(PI, logvar1, norm_value=norm_value);
    kl_div = calc_kl_divergence(mu1, logvar1, norm_value=norm_value);
    print('kl div uniform: ' + str(kl_div))
    denom = w_modalities[0]*torch.min(torch.Tensor([kl_div.exp(), 100000]));
    for k in range(0, len(mus)):
        if index == k:
            denom += w_modalities[k+1];
        else:
            kl_div = calc_kl_divergence(mu1, logvar1, mus[k], logvars[k], norm_value=norm_value)
            print('kl div ' + str(k) + ': ' + str(kl_div))
            denom += w_modalities[k+1]*torch.min(torch.Tensor([kl_div.exp(), 100000]));
    ub = torch.log(nom) - torch.log(denom) + entropy;
    return ub;


def calc_entropy_gauss(flags, logvar, norm_value=None):
    PI = torch.Tensor([math.pi]);
    if flags.cuda:
        PI = PI.cuda();
    ent = 0.5*torch.sum(torch.log(2*PI) + logvar + 1)
    if norm_value is not None:
        ent = ent / norm_value;
    return ent;