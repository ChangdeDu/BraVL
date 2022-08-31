
import torch
import torch.nn as nn


class EncoderText(nn.Module):
    def __init__(self, flags):
        super(EncoderText, self).__init__()
        self.flags = flags;
        self.hidden_dim = 256;

        modules = []
        modules.append(nn.Sequential(nn.Linear(flags.m3_dim, self.hidden_dim), nn.ReLU(True)))
        modules.extend([nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU(True))
                        for _ in range(flags.num_hidden_layers - 1)])
        self.enc = nn.Sequential(*modules)
        self.relu = nn.ReLU();
        self.hidden_mu = nn.Linear(in_features=self.hidden_dim, out_features=flags.class_dim, bias=True)
        self.hidden_logvar = nn.Linear(in_features=self.hidden_dim, out_features=flags.class_dim, bias=True)


    def forward(self, x):
        h = self.enc(x);
        h = h.view(h.size(0), -1);
        latent_space_mu = self.hidden_mu(h);
        latent_space_logvar = self.hidden_logvar(h);
        latent_space_mu = latent_space_mu.view(latent_space_mu.size(0), -1);
        latent_space_logvar = latent_space_logvar.view(latent_space_logvar.size(0), -1);
        return None, None, latent_space_mu, latent_space_logvar;



class DecoderText(nn.Module):
    def __init__(self, flags):
        super(DecoderText, self).__init__();
        self.flags = flags;
        self.hidden_dim = 256;
        modules = []

        modules.append(nn.Sequential(nn.Linear(flags.class_dim, self.hidden_dim), nn.ReLU(True)))

        modules.extend([nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU(True))
                        for _ in range(flags.num_hidden_layers - 1)])
        self.dec = nn.Sequential(*modules)
        self.fc3 = nn.Linear(self.hidden_dim, flags.m3_dim)
        self.relu = nn.ReLU();


    def forward(self, style_latent_space, class_latent_space):
        z = class_latent_space;
        x_hat = self.dec(z);
        x_hat = self.fc3(x_hat);
        return x_hat, torch.tensor(0.75).to(z.device);