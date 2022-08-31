import os
import torch

# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    # Print New Line on Complete
    if iteration == total:
        print()

def get_likelihood(str):
    if str == 'laplace':
        pz = dist.Laplace
    elif str == 'bernoulli':
        pz = dist.Bernoulli
    elif str == 'normal':
        pz = dist.Normal
    elif str == 'categorical':
        pz = dist.OneHotCategorical
    else:
        print('likelihood not implemented')
        pz = None
    return pz


def reweight_weights(w):
    w = w / w.sum()
    return w


def mixture_component_selection(flags, mus, logvars, w_modalities=None):
    #if not defined, take pre-defined weights
    num_components = mus.shape[0]
    num_samples = mus.shape[1]
    if w_modalities is None:
        w_modalities = torch.Tensor(flags.alpha_modalities).to(flags.device)
    idx_start = []
    idx_end = []
    for k in range(0, num_components):
        if k == 0:
            i_start = 0
        else:
            i_start = int(idx_end[k-1])
        if k == w_modalities.shape[0]-1:
            i_end = num_samples
        else:
            i_end = i_start + int(torch.floor(num_samples*w_modalities[k]))
        idx_start.append(i_start)
        idx_end.append(i_end)
    idx_end[-1] = num_samples
    mu_sel = torch.cat([mus[k, idx_start[k]:idx_end[k], :] for k in range(w_modalities.shape[0])])
    logvar_sel = torch.cat([logvars[k, idx_start[k]:idx_end[k], :] for k in range(w_modalities.shape[0])])
    return [mu_sel, logvar_sel]


def calc_elbo(exp, modality, recs, klds):
    flags = exp.flags
    mods = exp.modalities
    s_weights = exp.style_weights
    r_weights = exp.rec_weights
    kld_content = klds['content']
    if modality == 'joint':
        w_style_kld = 0.0
        w_rec = 0.0
        klds_style = klds['style']
        for k, m_key in enumerate(mods.keys()):
                w_style_kld += s_weights[m_key] * klds_style[m_key]
                w_rec += r_weights[m_key] * recs[m_key]
        kld_style = w_style_kld
        rec_error = w_rec
    else:
        beta_style_mod = s_weights[modality]
        #rec_weight_mod = r_weights[modality]
        rec_weight_mod = 1.0
        kld_style = beta_style_mod * klds['style'][modality]
        rec_error = rec_weight_mod * recs[modality]
    div = flags.beta_content * kld_content + flags.beta_style * kld_style
    elbo = rec_error + flags.beta * div
    return elbo


def save_and_log_flags(flags):
    #filename_flags = os.path.join(flags.dir_experiment_run, 'flags.json')
    #with open(filename_flags, 'w') as f:
    #    json.dump(flags.__dict__, f, indent=2, sort_keys=True)

    filename_flags_rar = os.path.join(flags.dir_experiment_run, 'flags.rar')
    torch.save(flags, filename_flags_rar)
    str_args = ''
    for k, key in enumerate(sorted(flags.__dict__.keys())):
        str_args = str_args + '\n' + key + ': ' + str(flags.__dict__[key])
    return str_args


class Flatten(torch.nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Unflatten(torch.nn.Module):
    def __init__(self, ndims):
        super(Unflatten, self).__init__()
        self.ndims = ndims

    def forward(self, x):
        return x.view(x.size(0), *self.ndims)
