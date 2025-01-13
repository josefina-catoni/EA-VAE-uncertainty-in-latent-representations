import sys, os
import random
import numpy as np

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm.auto import tqdm
from torch.utils.data import Dataset
from torch.nn.utils import clip_grad_value_

def set_seed(seed=None, seed_torch=True):
  """
  Function that controls randomness. NumPy and random modules must be imported.

  Args:
    seed : Integer
      A non-negative integer that defines the random state. Default is `None`.
    seed_torch : Boolean
      If `True` sets the random seed for pytorch tensors, so pytorch module
      must be imported. Default is `True`.

  Returns:
    Nothing.
  """
  if seed is None:
    seed = np.random.choice(2 ** 32)
  random.seed(seed)
  np.random.seed(seed)
  if seed_torch:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

  print(f'Random seed {seed} has been set.')

class CsikorDataset(Dataset):
    def __init__(self, labels, images, transform=None, target_transform=None):
        self.labels = labels
        self.images = images
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


class CsikorShuffDataset(Dataset):
    def __init__(self, labels, images, transform=None, target_transform=None):
        self.labels = labels
        self.images = images
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
    
class Laplace_FC_VAE_Encoder(nn.Module):
    def __init__(self,latent_size,input_size):
        super(Laplace_FC_VAE_Encoder, self).__init__()
        self.latent_size = latent_size
        self.input_size = input_size
        self.fc_1 = nn.Linear(in_features=self.input_size, out_features=2000)
        self.fc_2 = nn.Linear(in_features=2000, out_features=2000)
        self.fc_mu = nn.Linear(in_features=2000, out_features=self.latent_size)           
        self.fc_logvar = nn.Linear(in_features=2000, out_features=self.latent_size) 
    
    def forward(self, x):
        x = F.softplus(self.fc_1(x))
        x = F.softplus(self.fc_2(x))
        x_mu = self.fc_mu(x)
        x_logvar = self.fc_logvar(x)
        
        return x_mu, x_logvar 

class Laplace_FC_VAE_Decoder(nn.Module):
    def __init__(self,latent_size,input_size):
        super(Laplace_FC_VAE_Decoder, self).__init__()
        self.latent_size = latent_size
        self.input_size = input_size
        self.lin = nn.Linear(in_features=self.latent_size, out_features=self.input_size)
        
    def forward(self, x):
        x = self.lin(x)
        return x

class Laplace_FC_VAE(nn.Module):
    def __init__(self, ntrain, nval, latent_size, input_size, l_rate, w_decay, device):
        super(Laplace_FC_VAE, self).__init__()
        self.ntrain = ntrain
        self.nval = nval        
        self.latent_size = latent_size
        self.input_size = input_size
        self.encoder = Laplace_FC_VAE_Encoder(self.latent_size,self.input_size)
        self.decoder = Laplace_FC_VAE_Decoder(self.latent_size,self.input_size)
        self.optimizer = torch.optim.Adam(params=self.parameters(), lr=l_rate, weight_decay=w_decay)
        self.device = device

    def forward(self, x, only_mu=False):
        latent_mu, latent_logvar = self.encoder(x)
        latent = self.latent_sample(latent_mu, latent_logvar, only_mu)
        x_recon = self.decoder(latent)
        return x_recon, latent_mu, latent_logvar

    def latent_sample(self, mu, logvar, only_mu:False):
        if only_mu:
            return mu 
        else:
            # the reparameterization trick
            b = (logvar.exp()*0.5).sqrt()
            bound = 0.499
            u = torch.empty_like(b).uniform_(-bound, bound)
            eps = -u.sign().mul(torch.log(1-2*u.abs()))
            return eps.mul(b).add_(mu)
    
    def kl_div_unit_prior(self, mu, logvar):
        delta_mu = mu.abs()
        logb = 0.5 * (logvar - torch.log(torch.tensor(2.0))) 
        logb_prior = - 0.5 * torch.log(torch.tensor(2.0))
        delta_logb = logb_prior - logb
        b = logb.exp()
        b_prior = logb_prior.exp()

        kldivergence = torch.sum(- 1 + delta_logb + torch.div(delta_mu + b.mul(torch.div(-delta_mu,b).exp()),b_prior))
        
        return kldivergence
    
    def kl_div_cust_prior(self, mu, logvar, mu_prior, logvar_prior):
        delta_mu = (mu - mu_prior).abs()
        logb = 0.5 * (logvar - torch.log(torch.tensor(2.0))) 
        logb_prior = 0.5 * (logvar_prior - torch.log(torch.tensor(2.0)))
        delta_logb = logb_prior - logb
        b = logb.exp()
        b_prior = logb_prior.exp()

        kldivergence = torch.sum(- 1 + delta_logb + torch.div(delta_mu + b.mul(torch.div(-delta_mu,b).exp()),b_prior))

        return kldivergence
    
    def vae_loss(self, x, x_recon, mu, logvar):
        recon_loss = F.mse_loss(x_recon, x, reduction='sum')
        kl_div = self.kl_div_unit_prior(mu, logvar)
        
        return recon_loss, kl_div
        
    def vae_loss_wb_unit_prior(self, x, x_recon, mu, logvar):
        n_batch = x.shape[0]
        av_post_mu, av_post_logvar = mu.mean(dim=0) , torch.log((logvar.exp()).mean(dim=0) + mu.var(dim=0))
        recon_loss = F.mse_loss(x_recon, x, reduction='sum')
        kl_div_beta = self.kl_div_unit_prior(mu, logvar)
        kl_div_alpha = n_batch * self.kl_div_unit_prior(av_post_mu, av_post_logvar)
        
        return recon_loss, kl_div_beta, kl_div_alpha
    
    def vae_loss_wb_blank_prior(self, x, x_recon, mu, logvar):
        n_batch = x.shape[0]
        av_post_mu, av_post_logvar = mu.mean(dim=0).detach() , torch.log((logvar.exp()).mean(dim=0) + mu.var(dim=0)).detach()
        tgt_prior_mu, tgt_prior_logvar = self.encoder(torch.unsqueeze(x[0]*0.0,0))
        recon_loss = F.mse_loss(x_recon, x, reduction='sum')
        kl_div_beta = self.kl_div_cust_prior(mu, logvar, tgt_prior_mu, tgt_prior_logvar)
        kl_div_alpha = n_batch * self.kl_div_cust_prior(av_post_mu, av_post_logvar, torch.squeeze(tgt_prior_mu,0), torch.squeeze(tgt_prior_logvar,0))
        kl_div_delta = n_batch * self.kl_div_unit_prior(tgt_prior_mu[0], tgt_prior_logvar[0])
        
        return recon_loss, kl_div_beta, kl_div_alpha, kl_div_delta

class Laplace_FC_VAE_Encoder_w_contrast(nn.Module):
    def __init__(self,latent_size,input_size):
        super(Laplace_FC_VAE_Encoder_w_contrast, self).__init__()
        self.latent_size = latent_size
        self.input_size = input_size
        self.fc_1 = nn.Linear(in_features=self.input_size, out_features=2000)
        self.fc_2 = nn.Linear(in_features=2000, out_features=2000)
        self.fc_mu = nn.Linear(in_features=2000, out_features=self.latent_size)           
        self.fc_logvar = nn.Linear(in_features=2000, out_features=self.latent_size)
        self.var_pull = nn.Parameter(0.01*torch.empty_like(torch.ones(1)).normal_(), requires_grad=True)
    
    def estimate_contrast(self, x):
        return torch.sqrt(F.relu(torch.var(x,dim=1)-self.var_pull**2))    
    
    def forward(self, x):
        contrast_estimate = self.estimate_contrast(x)
        x = F.softplus(self.fc_1(x))
        x = F.softplus(self.fc_2(x))
        x_mu = self.fc_mu(x)
        x_logvar = self.fc_logvar(x)
        
        return x_mu, x_logvar, contrast_estimate

class Laplace_FC_VAE_Decoder_w_contrast(nn.Module):
    def __init__(self,latent_size,input_size):
        super(Laplace_FC_VAE_Decoder_w_contrast, self).__init__()
        self.latent_size = latent_size
        self.input_size = input_size
        self.lin = nn.Linear(in_features=self.latent_size, out_features=self.input_size)
        
    def forward(self, y, z):
        x = self.lin(y)*torch.unsqueeze(z,1)
        return x

class Laplace_FC_VAE_w_contrast(nn.Module):
    def __init__(self, ntrain, nval, latent_size, input_size, l_rate, w_decay, device):
        super(Laplace_FC_VAE_w_contrast, self).__init__()
        self.ntrain = ntrain
        self.nval = nval        
        self.latent_size = latent_size
        self.input_size = input_size
        self.encoder = Laplace_FC_VAE_Encoder_w_contrast(self.latent_size,self.input_size)
        self.decoder = Laplace_FC_VAE_Decoder_w_contrast(self.latent_size,self.input_size)
        self.optimizer = torch.optim.Adam(params=self.parameters(), lr=l_rate, weight_decay=w_decay)
        self.device = device
        
    def forward(self, x, only_mu=False):
        latent_mu, latent_logvar, contrast_estimate = self.encoder(x)
        latent = self.latent_sample(latent_mu, latent_logvar, only_mu)
        x_recon = self.decoder(latent, contrast_estimate)
        return x_recon, latent_mu, latent_logvar

    def latent_sample(self, mu, logvar, only_mu:False):
        if only_mu:
            return mu 
        else:
            # the reparameterization trick
            b = (logvar.exp()*0.5).sqrt()
            bound = 0.499
            u = torch.empty_like(b).uniform_(-bound, bound)
            eps = -u.sign().mul(torch.log(1-2*u.abs()))
            return eps.mul(b).add_(mu)
    
    def kl_div_unit_prior(self, mu, logvar):
        delta_mu = mu.abs()
        logb = 0.5 * (logvar - torch.log(torch.tensor(2.0))) 
        logb_prior = - 0.5 * torch.log(torch.tensor(2.0))
        delta_logb = logb_prior - logb
        b = logb.exp()
        b_prior = logb_prior.exp()

        kldivergence = torch.sum(- 1 + delta_logb + torch.div(delta_mu + b.mul(torch.div(-delta_mu,b).exp()),b_prior))
        
        return kldivergence
    
    def kl_div_cust_prior(self, mu, logvar, mu_prior, logvar_prior):
        delta_mu = (mu - mu_prior).abs()
        logb = 0.5 * (logvar - torch.log(torch.tensor(2.0))) 
        logb_prior = 0.5 * (logvar_prior - torch.log(torch.tensor(2.0)))
        delta_logb = logb_prior - logb
        b = logb.exp()
        b_prior = logb_prior.exp()

        kldivergence = torch.sum(- 1 + delta_logb + torch.div(delta_mu + b.mul(torch.div(-delta_mu,b).exp()),b_prior))

        return kldivergence
    
    def vae_loss(self, x, x_recon, mu, logvar):
        recon_loss = F.mse_loss(x_recon, x, reduction='sum')
        kl_div = self.kl_div_unit_prior(mu, logvar)
        
        return recon_loss, kl_div
        
    def vae_loss_wb_unit_prior(self, x, x_recon, mu, logvar):
        n_batch = x.shape[0]
        av_post_mu, av_post_logvar = mu.mean(dim=0) , torch.log((logvar.exp()).mean(dim=0) + mu.var(dim=0))
        recon_loss = F.mse_loss(x_recon, x, reduction='sum')
        kl_div_beta = self.kl_div_unit_prior(mu, logvar)
        kl_div_alpha = n_batch * self.kl_div_unit_prior(av_post_mu, av_post_logvar)
        
        return recon_loss, kl_div_beta, kl_div_alpha
    
    def vae_loss_wb_blank_prior(self, x, x_recon, mu, logvar):
        n_batch = x.shape[0]
        av_post_mu, av_post_logvar = mu.mean(dim=0).detach() , torch.log((logvar.exp()).mean(dim=0) + mu.var(dim=0)).detach()
        tgt_prior_mu, tgt_prior_logvar = self.encoder(torch.unsqueeze(x[0]*0.0,0))
        recon_loss = F.mse_loss(x_recon, x, reduction='sum')
        kl_div_beta = self.kl_div_cust_prior(mu, logvar, tgt_prior_mu, tgt_prior_logvar)
        kl_div_alpha = n_batch * self.kl_div_cust_prior(av_post_mu, av_post_logvar, torch.squeeze(tgt_prior_mu,0), torch.squeeze(tgt_prior_logvar,0))
        kl_div_delta = n_batch * self.kl_div_unit_prior(tgt_prior_mu[0], tgt_prior_logvar[0])
        
        return recon_loss, kl_div_beta, kl_div_alpha, kl_div_delta

class Gamma_Laplace_FC_VAE_Encoder(nn.Module):
    def __init__(self,latent_size,input_size):
        super(Gamma_Laplace_FC_VAE_Encoder, self).__init__()
        self.latent_size = latent_size
        self.input_size = input_size
        self.fc_1 = nn.Linear(in_features=self.input_size, out_features=2000)
        self.fc_2 = nn.Linear(in_features=2000, out_features=2000)
        self.fc_mu = nn.Linear(in_features=2000, out_features=self.latent_size)           
        self.fc_logvar = nn.Linear(in_features=2000, out_features=self.latent_size)
        self.fc_std = nn.Linear(in_features=1, out_features=20)
        self.fc_logtheta = nn.Linear(in_features=20, out_features=1)
    
    def forward(self, x):
        std = torch.std(x,dim=1).unsqueeze(1)
        std = F.softplus(self.fc_std(std))
        z_logtheta = self.fc_logtheta(std)
        x = F.softplus(self.fc_1(x))
        x = F.softplus(self.fc_2(x))
        x_mu = self.fc_mu(x)
        x_logvar = self.fc_logvar(x)
        
        return x_mu, x_logvar, z_logtheta
    
    def infer_logtheta(self, s):
        s_aux = F.softplus(self.fc_std(s.unsqueeze(1)))
        c_logtheta = self.fc_logtheta(s_aux)
        return c_logtheta


class Gamma_Laplace_FC_VAE_Decoder(nn.Module):
    def __init__(self,latent_size,input_size):
        super(Gamma_Laplace_FC_VAE_Decoder, self).__init__()
        self.latent_size = latent_size
        self.input_size = input_size
        self.lin = nn.Linear(in_features=self.latent_size, out_features=self.input_size)
        
    def forward(self, y, z):
        x = z*self.lin(y)
        return x

class Gamma_Laplace_FC_VAE(nn.Module):
    def __init__(self, ntrain, nval, latent_size, input_size, l_rate, w_decay, k_value,theta_prior, device):
        super(Gamma_Laplace_FC_VAE, self).__init__()
        self.ntrain = ntrain
        self.nval = nval        
        self.latent_size = latent_size
        self.input_size = input_size
        self.contrast_k = k_value
        self.contrast_theta_prior = torch.tensor(theta_prior)
        self.contrast_logtheta_prior = self.contrast_theta_prior.log()
        self.gamma_dist = torch.distributions.gamma.Gamma(self.contrast_k, 1)
        self.encoder = Gamma_Laplace_FC_VAE_Encoder(self.latent_size,self.input_size)
        self.decoder = Gamma_Laplace_FC_VAE_Decoder(self.latent_size,self.input_size)
        self.optimizer = torch.optim.Adam(params=self.parameters(), lr=l_rate, weight_decay=w_decay)
        self.device = device

    def forward(self, x, only_mu=False, contrast_mu=False,contrast_map=False):
        latent_mu, latent_logvar, contrast_logtheta = self.encoder(x)
        contrast = self.contrast_sample(contrast_logtheta, contrast_mu,contrast_map)
        latent = self.latent_sample(latent_mu, latent_logvar, only_mu)
        x_recon = self.decoder(latent, contrast)
        return x_recon, latent_mu, latent_logvar, contrast_logtheta

    def contrast_sample(self, logtheta, contrast_mu=False,contrast_map=False):
        theta = logtheta.exp()
        if contrast_mu:
            return self.contrast_k*theta
        elif contrast_map:
            return (self.contrast_k-1)*theta
        else:
            #u = -torch.empty((logtheta.size(dim=0),self.contrast_k),device=self.device).uniform_(0, 1)+1 #samples from uniform in (0,1]
            #w = -u.log().sum(dim=1).unsqueeze(1)
            w = torch.tensor([0])
            while len(w)!=len(theta):
                samps = self.gamma_dist.sample(sample_shape=(2*theta.size(0),1)).to(self.device)
                
                w = samps[(.001<samps) & (samps<6.5)][:len(theta)].unsqueeze(1)
            return w.mul(theta)
            

    def latent_sample(self, mu, logvar, only_mu=False):
        if only_mu:
            return mu 
        else:
            # the reparameterization trick
            b = (logvar.exp()*0.5).sqrt()
            bound = 0.499
            u = torch.empty_like(b).uniform_(-bound, bound)
            eps = -u.sign().mul(torch.log(1-2*u.abs()))
            return eps.mul(b).add_(mu)
    
    def kl_div_contrast(self,logtheta):
        theta = logtheta.exp()
        kldivergence = torch.sum(self.contrast_k * (self.contrast_logtheta_prior - logtheta + theta/self.contrast_theta_prior - 1))
        #kldivergence = torch.sum(self.contrast_k * (-1 - logtheta + theta))

        return kldivergence

    def kl_div_unit_prior(self, mu, logvar):
        delta_mu = mu.abs()
        logb = 0.5 * (logvar - torch.log(torch.tensor(2.0))) 
        logb_prior = - 0.5 * torch.log(torch.tensor(2.0))
        delta_logb = logb_prior - logb
        b = logb.exp()
        b_prior = logb_prior.exp()

        kldivergence = torch.sum(- 1 + delta_logb + torch.div(delta_mu + b.mul(torch.div(-delta_mu,b).exp()),b_prior))
        
        return kldivergence
    
    def kl_div_cust_prior(self, mu, logvar, mu_prior, logvar_prior):
        delta_mu = (mu - mu_prior).abs()
        logb = 0.5 * (logvar - torch.log(torch.tensor(2.0))) 
        logb_prior = 0.5 * (logvar_prior - torch.log(torch.tensor(2.0)))
        delta_logb = logb_prior - logb
        b = logb.exp()
        b_prior = logb_prior.exp()

        kldivergence = torch.sum(- 1 + delta_logb + torch.div(delta_mu + b.mul(torch.div(-delta_mu,b).exp()),b_prior))

        return kldivergence
    
    def vae_loss(self, x, x_recon, mu, logvar, logtheta):
        recon_loss = F.mse_loss(x_recon, x, reduction='sum')
        kl_div = self.kl_div_unit_prior(mu, logvar)
        kl_div_z = self.kl_div_contrast(logtheta)
        return recon_loss, kl_div, kl_div_z
        
    def vae_loss_wb_unit_prior(self, x, x_recon, mu, logvar):
        n_batch = x.shape[0]
        av_post_mu, av_post_logvar = mu.mean(dim=0) , torch.log((logvar.exp()).mean(dim=0) + mu.var(dim=0))
        recon_loss = F.mse_loss(x_recon, x, reduction='sum')
        kl_div_beta = self.kl_div_unit_prior(mu, logvar)
        kl_div_alpha = n_batch * self.kl_div_unit_prior(av_post_mu, av_post_logvar)
        
        return recon_loss, kl_div_beta, kl_div_alpha
    
    def vae_loss_wb_blank_prior(self, x, x_recon, mu, logvar):
        n_batch = x.shape[0]
        av_post_mu, av_post_logvar = mu.mean(dim=0).detach() , torch.log((logvar.exp()).mean(dim=0) + mu.var(dim=0)).detach()
        tgt_prior_mu, tgt_prior_logvar = self.encoder(torch.unsqueeze(x[0]*0.0,0))
        recon_loss = F.mse_loss(x_recon, x, reduction='sum')
        kl_div_beta = self.kl_div_cust_prior(mu, logvar, tgt_prior_mu, tgt_prior_logvar)
        kl_div_alpha = n_batch * self.kl_div_cust_prior(av_post_mu, av_post_logvar, torch.squeeze(tgt_prior_mu,0), torch.squeeze(tgt_prior_logvar,0))
        kl_div_delta = n_batch * self.kl_div_unit_prior(tgt_prior_mu[0], tgt_prior_logvar[0])
        
        return recon_loss, kl_div_beta, kl_div_alpha, kl_div_delta
    
    def infer_contrast(self, s, c_mu = False, c_map = False):
        c_logtheta = self.encoder.infer_logtheta(s)
        c_theta = c_logtheta.exp()
        if c_mu:
            return self.contrast_k*c_theta
        elif c_map:
            return (self.contrast_k-1)*c_theta
        else:
            print('Invalid contrast sampling method. Define either c_mu or c_map as True.')
            return None
   
   

class Gamma_free_Laplace_FC_VAE_Encoder(nn.Module):
    def __init__(self,latent_size,input_size):
        super(Gamma_free_Laplace_FC_VAE_Encoder, self).__init__()
        self.latent_size = latent_size
        self.input_size = input_size
        self.fc_1 = nn.Linear(in_features=self.input_size, out_features=2000)
        self.fc_2 = nn.Linear(in_features=2000, out_features=2000)
        self.fc_mu = nn.Linear(in_features=2000, out_features=self.latent_size)           
        self.fc_logvar = nn.Linear(in_features=2000, out_features=self.latent_size)
        self.fc_explain = nn.Linear(in_features=2000, out_features=20)
        self.fc_logtheta = nn.Linear(in_features=20, out_features=1)
    
    def forward(self, x):
        
        x = F.softplus(self.fc_1(x))
        z = F.softplus(self.fc_explain(x))
        x = F.softplus(self.fc_2(x))
        x_mu = self.fc_mu(x)
        x_logvar = self.fc_logvar(x)
        z_logtheta = self.fc_logtheta(z)
        
        return x_mu, x_logvar, z_logtheta
    

class Gamma_free_Laplace_FC_VAE(nn.Module):
    def __init__(self, ntrain, nval, latent_size, input_size, l_rate, w_decay, k_value,theta_prior, device):
        super(Gamma_free_Laplace_FC_VAE, self).__init__()
        self.ntrain = ntrain
        self.nval = nval        
        self.latent_size = latent_size
        self.input_size = input_size
        self.z_k = k_value
        self.z_theta_prior = torch.tensor(theta_prior)
        self.z_logtheta_prior = self.z_theta_prior.log()
        self.gamma_dist = torch.distributions.gamma.Gamma(self.z_k, 1)
        self.encoder = Gamma_free_Laplace_FC_VAE_Encoder(self.latent_size,self.input_size)
        self.decoder = Gamma_Laplace_FC_VAE_Decoder(self.latent_size,self.input_size)
        self.optimizer = torch.optim.Adam(params=self.parameters(), lr=l_rate, weight_decay=w_decay)
        self.device = device

    def forward(self, x, only_mu=False, only_zmu=False,only_zmap=False):
        latent_ymu, latent_ylogvar, latent_zlogtheta = self.encoder(x)
        latent_z = self.latent_zsample(latent_zlogtheta, only_zmu,only_zmap)
        latent_y = self.latent_ysample(latent_ymu, latent_ylogvar, only_mu)
        x_recon = self.decoder(latent_y, latent_z)
        return x_recon, latent_ymu, latent_ylogvar, latent_zlogtheta

    def latent_zsample(self, logtheta, only_zmu=False,only_zmap=False):
        theta = logtheta.exp()
        if only_zmu:
            return self.z_k*theta
        elif only_zmap:
            return (self.z_k-1)*theta
        else:
            #u = -torch.empty((logtheta.size(dim=0),self.z_k),device=self.device).uniform_(0, 1)+1 #samples from uniform in (0,1]
            #w = -u.log().sum(dim=1).unsqueeze(1)
            w = torch.tensor([0])
            while len(w)!=len(theta):
                samps = self.gamma_dist.sample(sample_shape=(2*theta.size(0),1)).to(self.device)
                
                w = samps[(.001<samps) & (samps<6.5)][:len(theta)].unsqueeze(1)
            return w.mul(theta)
            

    def latent_ysample(self, mu, logvar, only_mu=False):
        if only_mu:
            return mu 
        else:
            # the reparameterization trick
            b = (logvar.exp()*0.5).sqrt()
            bound = 0.499
            u = torch.empty_like(b).uniform_(-bound, bound)
            eps = -u.sign().mul(torch.log(1-2*u.abs()))
            return eps.mul(b).add_(mu)
    
    def kl_div_z(self,zlogtheta):
        ztheta = zlogtheta.exp()
        kldivergence = torch.sum(self.z_k * (self.z_logtheta_prior - zlogtheta + ztheta/self.z_theta_prior - 1))
        #kldivergence = torch.sum(self.z_k * (-1 - logtheta + theta))

        return kldivergence

    def kl_div_unit_prior(self, mu, logvar):
        delta_mu = mu.abs()
        logb = 0.5 * (logvar - torch.log(torch.tensor(2.0))) 
        logb_prior = - 0.5 * torch.log(torch.tensor(2.0))
        delta_logb = logb_prior - logb
        b = logb.exp()
        b_prior = logb_prior.exp()

        kldivergence = torch.sum(- 1 + delta_logb + torch.div(delta_mu + b.mul(torch.div(-delta_mu,b).exp()),b_prior))
        
        return kldivergence
    
    def kl_div_cust_prior(self, mu, logvar, mu_prior, logvar_prior):
        delta_mu = (mu - mu_prior).abs()
        logb = 0.5 * (logvar - torch.log(torch.tensor(2.0))) 
        logb_prior = 0.5 * (logvar_prior - torch.log(torch.tensor(2.0)))
        delta_logb = logb_prior - logb
        b = logb.exp()
        b_prior = logb_prior.exp()

        kldivergence = torch.sum(- 1 + delta_logb + torch.div(delta_mu + b.mul(torch.div(-delta_mu,b).exp()),b_prior))

        return kldivergence
    
    def vae_loss(self, x, x_recon, mu, logvar, logtheta):
        recon_loss = F.mse_loss(x_recon, x, reduction='sum')
        kl_div_y = self.kl_div_unit_prior(mu, logvar)
        kl_div_z = self.kl_div_z(logtheta)
        return recon_loss, kl_div_y, kl_div_z
        
    def vae_loss_wb_unit_prior(self, x, x_recon, mu, logvar):
        n_batch = x.shape[0]
        av_post_mu, av_post_logvar = mu.mean(dim=0) , torch.log((logvar.exp()).mean(dim=0) + mu.var(dim=0))
        recon_loss = F.mse_loss(x_recon, x, reduction='sum')
        kl_div_beta = self.kl_div_unit_prior(mu, logvar)
        kl_div_alpha = n_batch * self.kl_div_unit_prior(av_post_mu, av_post_logvar)
        
        return recon_loss, kl_div_beta, kl_div_alpha
    
    def vae_loss_wb_blank_prior(self, x, x_recon, mu, logvar):
        n_batch = x.shape[0]
        av_post_mu, av_post_logvar = mu.mean(dim=0).detach() , torch.log((logvar.exp()).mean(dim=0) + mu.var(dim=0)).detach()
        tgt_prior_mu, tgt_prior_logvar = self.encoder(torch.unsqueeze(x[0]*0.0,0))
        recon_loss = F.mse_loss(x_recon, x, reduction='sum')
        kl_div_beta = self.kl_div_cust_prior(mu, logvar, tgt_prior_mu, tgt_prior_logvar)
        kl_div_alpha = n_batch * self.kl_div_cust_prior(av_post_mu, av_post_logvar, torch.squeeze(tgt_prior_mu,0), torch.squeeze(tgt_prior_logvar,0))
        kl_div_delta = n_batch * self.kl_div_unit_prior(tgt_prior_mu[0], tgt_prior_logvar[0])
        
        return recon_loss, kl_div_beta, kl_div_alpha, kl_div_delta
    
    def infer_contrast(self, s, c_mu = False, c_map = False):
        c_logtheta = self.encoder.infer_logtheta(s)
        c_theta = c_logtheta.exp()
        if c_mu:
            return self.z_k*c_theta
        elif c_map:
            return (self.z_k-1)*c_theta
        else:
            print('Invalid contrast sampling method. Define either c_mu or c_map as True.')
            return None
        

class Laplace_FC_DVAE(nn.Module): #DenoiseVAE
    def __init__(self, ntrain, nval, latent_size, input_size, l_rate, w_decay, n_level, device):
        super(Laplace_FC_DVAE, self).__init__()
        self.ntrain = ntrain
        self.nval = nval        
        self.latent_size = latent_size
        self.input_size = input_size
        self.encoder = Laplace_FC_VAE_Encoder(self.latent_size,self.input_size)
        self.decoder = Laplace_FC_VAE_Decoder(self.latent_size,self.input_size)
        self.optimizer = torch.optim.Adam(params=self.parameters(), lr=l_rate, weight_decay=w_decay)
        self.noise_level = n_level
        self.device = device

        
    def forward(self, x, only_mu=False, return_noisy_x = False):
        noisy_x = x + self.noise_level*torch.empty_like(x).normal_()
        latent_mu, latent_logvar = self.encoder(noisy_x)
        latent = self.latent_sample(latent_mu, latent_logvar, only_mu)
        x_recon = self.decoder(latent)
        if return_noisy_x:
            return noisy_x, x_recon, latent_mu, latent_logvar
        else:
            return x_recon, latent_mu, latent_logvar

    def latent_sample(self, mu, logvar, only_mu:False):
        if only_mu:
            return mu 
        else:
            # the reparameterization trick
            b = (logvar.exp()*0.5).sqrt()
            bound = 0.499
            u = torch.empty_like(b).uniform_(-bound, bound)
            eps = -u.sign().mul(torch.log(1-2*u.abs()))
            return eps.mul(b).add_(mu)
    
    def kl_div_unit_prior(self, mu, logvar):
        delta_mu = mu.abs()
        logb = 0.5 * (logvar - torch.log(torch.tensor(2.0))) 
        logb_prior = - 0.5 * torch.log(torch.tensor(2.0))
        delta_logb = logb_prior - logb
        b = logb.exp()
        b_prior = logb_prior.exp()

        kldivergence = torch.sum(- 1 + delta_logb + torch.div(delta_mu + b.mul(torch.div(-delta_mu,b).exp()),b_prior))
        
        return kldivergence
    
    def kl_div_cust_prior(self, mu, logvar, mu_prior, logvar_prior):
        delta_mu = (mu - mu_prior).abs()
        logb = 0.5 * (logvar - torch.log(torch.tensor(2.0))) 
        logb_prior = 0.5 * (logvar_prior - torch.log(torch.tensor(2.0)))
        delta_logb = logb_prior - logb
        b = logb.exp()
        b_prior = logb_prior.exp()

        kldivergence = torch.sum(- 1 + delta_logb + torch.div(delta_mu + b.mul(torch.div(-delta_mu,b).exp()),b_prior))

        return kldivergence
    
    def vae_loss(self, x, x_recon, mu, logvar):
        recon_loss = F.mse_loss(x_recon, x, reduction='sum')
        kl_div = self.kl_div_unit_prior(mu, logvar)
        
        return recon_loss, kl_div
        
    def vae_loss_wb_unit_prior(self, x, x_recon, mu, logvar):
        n_batch = x.shape[0]
        av_post_mu, av_post_logvar = mu.mean(dim=0) , torch.log((logvar.exp()).mean(dim=0) + mu.var(dim=0))
        recon_loss = F.mse_loss(x_recon, x, reduction='sum')
        kl_div_beta = self.kl_div_unit_prior(mu, logvar)
        kl_div_alpha = n_batch * self.kl_div_unit_prior(av_post_mu, av_post_logvar)
        
        return recon_loss, kl_div_beta, kl_div_alpha
    
    def vae_loss_wb_blank_prior(self, x, x_recon, mu, logvar):
        n_batch = x.shape[0]
        av_post_mu, av_post_logvar = mu.mean(dim=0).detach() , torch.log((logvar.exp()).mean(dim=0) + mu.var(dim=0)).detach()
        tgt_prior_mu, tgt_prior_logvar = self.encoder(torch.unsqueeze(x[0]*0.0,0))
        recon_loss = F.mse_loss(x_recon, x, reduction='sum')
        kl_div_beta = self.kl_div_cust_prior(mu, logvar, tgt_prior_mu, tgt_prior_logvar)
        kl_div_alpha = n_batch * self.kl_div_cust_prior(av_post_mu, av_post_logvar, torch.squeeze(tgt_prior_mu,0), torch.squeeze(tgt_prior_logvar,0))
        kl_div_delta = n_batch * self.kl_div_unit_prior(tgt_prior_mu[0], tgt_prior_logvar[0])
        
        return recon_loss, kl_div_beta, kl_div_alpha, kl_div_delta


class Normal_FC_VAE_Encoder(nn.Module):
    def __init__(self,latent_size,input_size):
        super(Normal_FC_VAE_Encoder, self).__init__()
        self.latent_size = latent_size
        self.input_size = input_size
        self.fc_1 = nn.Linear(in_features=self.input_size, out_features=2000)
        self.fc_2 = nn.Linear(in_features=2000, out_features=2000)
        self.fc_mu = nn.Linear(in_features=2000, out_features=self.latent_size)           
        self.fc_logvar = nn.Linear(in_features=2000, out_features=self.latent_size) 
    
    def forward(self, x):
        x = F.softplus(self.fc_1(x))
        x = F.softplus(self.fc_2(x))
        x_mu = self.fc_mu(x)
        x_logvar = self.fc_logvar(x)
        
        return x_mu, x_logvar 

class Normal_FC_VAE_Decoder(nn.Module):
    def __init__(self,latent_size,input_size):
        super(Normal_FC_VAE_Decoder, self).__init__()
        self.latent_size = latent_size
        self.input_size = input_size
        self.lin = nn.Linear(in_features=self.latent_size, out_features=self.input_size)
        
  
    def forward(self, x):
        x = self.lin(x)
        return x
    
class Normal_FC_VAE(nn.Module):
    def __init__(self, ntrain, nval, latent_size, input_size, l_rate, w_decay, device):
        super(Normal_FC_VAE, self).__init__()
        self.ntrain = ntrain
        self.nval = nval
        self.latent_size = latent_size
        self.input_size = input_size
        self.encoder = Normal_FC_VAE_Encoder(self.latent_size,self.input_size)
        self.decoder = Normal_FC_VAE_Decoder(self.latent_size,self.input_size)
        self.optimizer = torch.optim.Adam(params=self.parameters(), lr=l_rate, weight_decay=w_decay)
        self.device = device

        
    def forward(self, x, only_mu=False):
        latent_mu, latent_logvar = self.encoder(x)
        latent = self.latent_sample(latent_mu, latent_logvar, only_mu)
        x_recon = self.decoder(latent)
        return x_recon, latent_mu, latent_logvar 

    def latent_sample(self, mu, logvar, only_mu:False):
        if only_mu:
            return mu 
        else:
            # the reparameterization trick
            std = logvar.mul(0.5).exp_()
            eps = torch.empty_like(std).normal_()
            return eps.mul(std).add_(mu)
        
    def kl_div_unit_prior(self, mu, logvar):
        var_inv_var = logvar.exp()
        log_ratio_vars = -logvar
        weighted_dmu2 = mu.pow(2)

        kldivergence =  0.5 * torch.sum( -1 + log_ratio_vars + weighted_dmu2 + var_inv_var)
        
        return kldivergence
        
    def kl_div_cust_prior(self, mu, logvar, mu_prior, logvar_prior):
        var_inv_var = (logvar-logvar_prior).exp()
        log_ratio_vars = logvar_prior - logvar
        weighted_dmu2 = torch.div((mu-mu_prior).pow(2),logvar_prior.exp())

        kldivergence =  0.5 * torch.sum( -1 + log_ratio_vars + weighted_dmu2 + var_inv_var)
        return kldivergence
    
    def vae_loss(self, x, x_recon, mu, logvar):
        recon_loss = F.mse_loss(x_recon, x, reduction='sum')
        kl_div = self.kl_div_unit_prior(mu, logvar)
        
        return recon_loss, kl_div
    
    def vae_loss_wb_unit_prior(self, x, x_recon, mu, logvar):
        n_batch = x.shape[0]
        av_post_mu, av_post_logvar = mu.mean(dim=0) , torch.log((logvar.exp()).mean(dim=0) + mu.var(dim=0))
        recon_loss = F.mse_loss(x_recon, x, reduction='sum')
        kl_div_beta = self.kl_div_unit_prior(mu, logvar)
        kl_div_alpha = n_batch * self.kl_div_unit_prior(av_post_mu, av_post_logvar)
        
        return recon_loss, kl_div_beta, kl_div_alpha

    def vae_loss_wb_blank_prior(self, x, x_recon, mu, logvar):
        n_batch = x.shape[0]
        av_post_mu, av_post_logvar = mu.mean(dim=0).detach() , torch.log((logvar.exp()).mean(dim=0) + mu.var(dim=0)).detach()
        tgt_prior_mu, tgt_prior_logvar = self.encoder(torch.unsqueeze(x[0]*0.0,0))
        recon_loss = F.mse_loss(x_recon, x, reduction='sum')
        kl_div_beta = self.kl_div_cust_prior(mu, logvar, tgt_prior_mu, tgt_prior_logvar)
        kl_div_alpha = n_batch * self.kl_div_cust_prior(av_post_mu, av_post_logvar, torch.squeeze(tgt_prior_mu,0), torch.squeeze(tgt_prior_logvar,0))
        kl_div_delta = n_batch * self.kl_div_unit_prior(tgt_prior_mu[0], tgt_prior_logvar[0])
        
        return recon_loss, kl_div_beta, kl_div_alpha, kl_div_delta


def vae_cost_function(recon_loss, kl_div, beta):
    loss = recon_loss + beta * kl_div
    return loss

def fit_step(model, x, beta):
    x_hat, latent_mu, latent_logvar = model(x)
    recon_loss, kl_div = model.vae_loss(x, x_hat, latent_mu, latent_logvar)
    loss = vae_cost_function(recon_loss, kl_div, beta)
    model.optimizer.zero_grad()
    loss.backward() 
    model.optimizer.step() 
    return loss.item(), recon_loss.item(), kl_div.item()

def train_and_val(model, train_dataloader, validation_dataloader, nepochs, variational_beta, saving_path):
    
    (train_loss_history, train_rec_loss_history, train_kl_div_history, val_loss_history, 
    val_rec_loss_history, val_kl_div_history) = [],[],[],[],[],[]
    
    epoch_best=0
    print("Training starts...")
    epoch_iterator = tqdm(range(nepochs))
    for epoch in epoch_iterator:

        model.train()
        loss_epoch_s, rec_loss_epoch_s, kl_div_epoch_s = 0., 0., 0.
        for batch in train_dataloader:            
            if type(batch) == list: 
                batch = batch[0]                     
            
            l_s, rl_s, kl_s  = fit_step(model,batch.to(model.device),variational_beta)
            loss_epoch_s += l_s
            rec_loss_epoch_s += rl_s
            kl_div_epoch_s += kl_s

        model.eval()
        val_loss_epoch, val_rec_loss_epoch, val_kl_div_epoch = 0., 0., 0.
        
        for val_batch in validation_dataloader:
            if type(val_batch) == list: 
                val_batch = val_batch[0]

            with torch.no_grad():   
                val_batch = val_batch.to(model.device)
                val_batch_recon_sample, val_mu_sample, val_logvar_sample= model(val_batch)
                
                vrl, vkl =  model.vae_loss(val_batch, val_batch_recon_sample, val_mu_sample, val_logvar_sample)
                vl = vae_cost_function(vrl, vkl, variational_beta)
                val_loss_epoch += vl.item()
                val_rec_loss_epoch += vrl.item()
                val_kl_div_epoch += vkl.item()

        #print(model.encoder.var_pull)

        val_loss_history.append(val_loss_epoch/model.nval)
        val_rec_loss_history.append(val_rec_loss_epoch/model.nval)
        val_kl_div_history.append(val_kl_div_epoch/model.nval)

        train_loss_history.append(loss_epoch_s/model.ntrain)
        train_rec_loss_history.append(rec_loss_epoch_s/model.ntrain)  
        train_kl_div_history.append(kl_div_epoch_s/model.ntrain)
                               
        if ((epoch == 0)or((val_loss_epoch/model.nval) < val_loss_history[epoch_best])):
            epoch_best = epoch
            torch.save(model.state_dict(), saving_path + f"bestnet.pth")
            torch.save(epoch_best, saving_path + f"bestepoch.pth") 
            torch.save(torch.tensor([train_loss_history, train_rec_loss_history, train_kl_div_history, val_loss_history, val_rec_loss_history, val_kl_div_history],requires_grad=False).T,saving_path +f"./bestloss.pth")     

        if (epoch+1)%100==0: #checkpoints every 100 epochs
            torch.save(model.state_dict(), saving_path +f"./checkpoint_epoch_{(epoch+1):04d}.pth")
            torch.save(torch.tensor([train_loss_history, train_rec_loss_history, train_kl_div_history, val_loss_history, val_rec_loss_history, val_kl_div_history],requires_grad=False).T,saving_path + f"finalloss.pth")
        
        epoch_iterator.set_postfix({'best-epoch': epoch_best,
                                    't-loss': train_loss_history[epoch_best],
                                    'v-loss': val_loss_history[epoch_best],
                                    't-rec-loss': train_rec_loss_history[epoch_best],
                                    'v-rec-loss': val_rec_loss_history[epoch_best],
                                    't-kl-div': train_kl_div_history[epoch_best],
                                    'v-kl-div': val_kl_div_history[epoch_best]}
                                    )
        
    torch.save(torch.tensor([train_loss_history, train_rec_loss_history, train_kl_div_history, val_loss_history, val_rec_loss_history, val_kl_div_history],requires_grad=False).T,saving_path + f"finalloss.pth")
    print('...training ended.')
    return


def vae_contrast_cost_function(recon_loss, kl_div, kl_div_z, beta, rho):
    loss = recon_loss + beta * kl_div + rho * kl_div_z
    return loss

def fit_step_contrast(model, x, beta, rho):
    x_hat, latent_ymu, latent_ylogvar, latent_zlogtheta = model(x)
    recon_loss, kl_div, kl_div_z = model.vae_loss(x, x_hat, latent_ymu, latent_ylogvar, latent_zlogtheta)
    loss = vae_contrast_cost_function(recon_loss, kl_div, kl_div_z, beta, rho)
    model.optimizer.zero_grad()
    loss.backward()
    clip_grad_value_(model.parameters(), 2e6)
    model.optimizer.step() 
    return loss.item(), recon_loss.item(), kl_div.item(), kl_div_z.item()

def z_train_and_val(model, train_dataloader, validation_dataloader, nepochs, variational_beta, variational_rho, saving_path):
    
    (train_loss_history, train_rec_loss_history, train_kl_div_history, train_kl_div_z_history, val_loss_history, 
    val_rec_loss_history, val_kl_div_history, val_kl_div_z_history) = [],[],[],[],[],[],[],[]
    
    epoch_best=0
    print("Training starts...")
    epoch_iterator = tqdm(range(nepochs))
    for epoch in epoch_iterator:

        model.train()
        loss_epoch_s, rec_loss_epoch_s, kl_div_epoch_s, kl_div_z_epoch_s = 0., 0., 0., 0.
        for batch in train_dataloader:            
            if type(batch) == list: 
                batch = batch[0]                     
            
            l_s, rl_s, kl_s, klz_s  = fit_step_contrast(model,batch.to(model.device),variational_beta,variational_rho)
            loss_epoch_s += l_s
            rec_loss_epoch_s += rl_s
            kl_div_epoch_s += kl_s
            kl_div_z_epoch_s += klz_s

        model.eval()
        val_loss_epoch, val_rec_loss_epoch, val_kl_div_epoch, val_kl_div_z_epoch = 0., 0., 0., 0.
        
        for val_batch in validation_dataloader:
            if type(val_batch) == list: 
                val_batch = val_batch[0]

            with torch.no_grad():   
                val_batch = val_batch.to(model.device)
                val_batch_recon_sample, val_mu_sample, val_logvar_sample, val_logtheta_sample= model(val_batch)
                
                vrl, vkl, vklz =  model.vae_loss(val_batch, val_batch_recon_sample, val_mu_sample, val_logvar_sample, val_logtheta_sample)
                vl = vae_contrast_cost_function(vrl, vkl, vklz, variational_beta, variational_rho)
                val_loss_epoch += vl.item()
                val_rec_loss_epoch += vrl.item()
                val_kl_div_epoch += vkl.item()
                val_kl_div_z_epoch += vklz.item()

        #print(model.encoder.var_pull)

        val_loss_history.append(val_loss_epoch/model.nval)
        val_rec_loss_history.append(val_rec_loss_epoch/model.nval)
        val_kl_div_history.append(val_kl_div_epoch/model.nval)
        val_kl_div_z_history.append(val_kl_div_z_epoch/model.nval)

        train_loss_history.append(loss_epoch_s/model.ntrain)
        train_rec_loss_history.append(rec_loss_epoch_s/model.ntrain)  
        train_kl_div_history.append(kl_div_epoch_s/model.ntrain)
        train_kl_div_z_history.append(kl_div_z_epoch_s/model.ntrain)
                               
        if ((epoch == 0)or((val_loss_epoch/model.nval) < val_loss_history[epoch_best])):
            epoch_best = epoch
            torch.save(model.state_dict(), saving_path + f"bestnet.pth")
            torch.save(epoch_best, saving_path + f"bestepoch.pth") 
            torch.save(torch.tensor([train_loss_history, train_rec_loss_history, train_kl_div_history, train_kl_div_z_history, val_loss_history, val_rec_loss_history, val_kl_div_history, val_kl_div_z_history],requires_grad=False).T,saving_path +f"./bestloss.pth")     

        if (epoch+1)%50==0: #checkpoints every 50 epohcs
            torch.save(model.state_dict(), saving_path +f"./checkpoint_epoch_{(epoch+1):04d}.pth")
            torch.save(torch.tensor([train_loss_history, train_rec_loss_history, train_kl_div_history, train_kl_div_z_history, val_loss_history, val_rec_loss_history, val_kl_div_history, val_kl_div_z_history],requires_grad=False).T,saving_path + f"finalloss.pth")
        
        epoch_iterator.set_postfix({'best-epoch': epoch_best,
                                    't-loss': train_loss_history[epoch_best],
                                    'v-loss': val_loss_history[epoch_best],
                                    't-rec-loss': train_rec_loss_history[epoch_best],
                                    'v-rec-loss': val_rec_loss_history[epoch_best],
                                    't-kl-div': train_kl_div_history[epoch_best],
                                    'v-kl-div': val_kl_div_history[epoch_best],
                                    't-kl-z-div': train_kl_div_z_history[epoch_best],
                                    'v-kl-z-div': val_kl_div_z_history[epoch_best]
                                    })
        
    torch.save(torch.tensor([train_loss_history, train_rec_loss_history, train_kl_div_history, train_kl_div_z_history, val_loss_history, val_rec_loss_history, val_kl_div_history, val_kl_div_z_history],requires_grad=False).T,saving_path + f"finalloss.pth")
    print('...training ended.')
    return


def vae_wb_unit_prior_cost_function(recon_loss, kl_div_beta, kl_div_alpha, beta, alpha):
    loss = recon_loss + beta * kl_div_beta + alpha * kl_div_alpha
    return loss

def fit_step_wb_unit_prior(model, x, beta, alpha):
    x_hat, latent_mu, latent_logvar = model(x)
    recon_loss, kl_div_beta, kl_div_alpha = model.vae_loss_wb_unit_prior(x, x_hat, latent_mu, latent_logvar)
    loss =  vae_wb_unit_prior_cost_function(recon_loss, kl_div_beta, kl_div_alpha, beta, alpha)
    model.optimizer.zero_grad()
    loss.backward() 
    model.optimizer.step() 
    return loss.item(), recon_loss.item(), kl_div_beta.item(), kl_div_alpha.item()

def train_and_val_wb_unit_prior(model, train_dataloader, validation_dataloader, nepochs, variational_beta, variational_alpha, saving_path):
    
    (train_loss_history, train_rec_loss_history, train_kl_div_b_history, train_kl_div_a_history,
     val_loss_history, val_rec_loss_history, val_kl_div_b_history, val_kl_div_a_history) = [],[],[],[],[],[],[],[]
    
    epoch_best=0
    print("Training starts...")
    epoch_iterator = tqdm(range(nepochs))
    for epoch in epoch_iterator:

        model.train()
        loss_epoch_s, rec_loss_epoch_s, kl_div_b_epoch_s, kl_div_a_epoch_s = 0, 0, 0, 0
        for batch in train_dataloader:            
            if type(batch) == list: 
                batch = batch[0]                     
            
            l_s, rl_s, klb_s, kla_s = fit_step_wb_unit_prior(model,batch.to(model.device), variational_beta, variational_alpha)
            loss_epoch_s += l_s
            rec_loss_epoch_s += rl_s
            kl_div_b_epoch_s += klb_s
            kl_div_a_epoch_s += kla_s

        model.eval()

        val_loss_epoch, val_rec_loss_epoch, val_kl_div_b_epoch, val_kl_div_a_epoch = 0, 0, 0, 0
        
        for val_batch in validation_dataloader:
            if type(val_batch) == list: 
                val_batch = val_batch[0]

            with torch.no_grad():   
                val_batch = val_batch.to(model.device)
                val_batch_recon_sample, val_mu_sample, val_logvar_sample= model(val_batch)

                vrl, vklb, vkla = model.vae_loss_wb_unit_prior(val_batch, val_batch_recon_sample, val_mu_sample, val_logvar_sample)
                vl = vae_wb_unit_prior_cost_function(vrl, vklb, vkla, variational_beta, variational_alpha)

                val_loss_epoch += vl.item()
                val_rec_loss_epoch += vrl.item()
                val_kl_div_b_epoch += vklb.item()
                val_kl_div_a_epoch += vkla.item()

        val_loss_history.append(val_loss_epoch/model.nval)
        val_rec_loss_history.append(val_rec_loss_epoch/model.nval)
        val_kl_div_b_history.append(val_kl_div_b_epoch/model.nval)
        val_kl_div_a_history.append(val_kl_div_a_epoch/model.nval)

        train_loss_history.append(loss_epoch_s/model.ntrain)
        train_rec_loss_history.append(rec_loss_epoch_s/model.ntrain)  
        train_kl_div_b_history.append(kl_div_b_epoch_s/model.ntrain)
        train_kl_div_a_history.append(kl_div_a_epoch_s/model.ntrain)
                        
        if ((epoch == 0)or((val_loss_epoch/model.nval) < val_loss_history[epoch_best])):
            epoch_best = epoch
            torch.save(model.state_dict(), saving_path + f"bestnet.pth")
            torch.save(epoch_best, saving_path + f"bestepoch.pth") 
            torch.save(torch.tensor([train_loss_history, train_rec_loss_history, train_kl_div_b_history, train_kl_div_a_history,val_loss_history, val_rec_loss_history, val_kl_div_b_history, val_kl_div_a_history],requires_grad=False).T,saving_path +f"./bestloss.pth")     

        if (epoch+1)%50==0: #checkpoints every 50 epohcs
            torch.save(model.state_dict(), saving_path +f"./checkpoint_epoch_{(epoch+1):04d}.pth")
        
        epoch_iterator.set_postfix({'best-epoch': epoch_best,
                                    't-loss': train_loss_history[epoch_best],
                                    'v-loss': val_loss_history[epoch_best],
                                    't-rec-loss': train_rec_loss_history[epoch_best],
                                    'v-rec-loss': val_rec_loss_history[epoch_best],
                                    't-kl-beta': train_kl_div_b_history[epoch_best],
                                    'v-kl-beta': val_kl_div_b_history[epoch_best],
                                    't-kl-alpha': train_kl_div_a_history[epoch_best],
                                    'v-kl-alpha': val_kl_div_a_history[epoch_best]}
                                    )
        
    torch.save(torch.tensor([train_loss_history, train_rec_loss_history, train_kl_div_b_history, train_kl_div_a_history,val_loss_history, val_rec_loss_history, val_kl_div_b_history, val_kl_div_a_history],requires_grad=False).T,saving_path + f"finalloss.pth")
    print('...training ended.')
    return


def vae_wb_blank_prior_cost_function(recon_loss, kl_div_beta, kl_div_alpha, kl_div_delta, beta, alpha, delta):
    loss = recon_loss + beta * kl_div_beta + alpha * kl_div_alpha + delta * kl_div_delta
    return loss

def fit_step_wb_blank_prior(model, x, beta, alpha, delta):
    x_hat, latent_mu, latent_logvar = model(x)
    recon_loss, kl_div_beta, kl_div_alpha, kl_div_delta = model.vae_loss_wb_blank_prior(x, x_hat, latent_mu, latent_logvar)
    loss =  vae_wb_blank_prior_cost_function(recon_loss, kl_div_beta, kl_div_alpha, kl_div_delta, beta, alpha, delta)
    model.optimizer.zero_grad()
    loss.backward() 
    model.optimizer.step() 
    return loss.item(), recon_loss.item(), kl_div_beta.item(), kl_div_alpha.item(), kl_div_delta.item()

def train_and_val_wb_blank_prior(model, train_dataloader, validation_dataloader, nepochs, variational_beta, variational_alpha, variational_delta, saving_path):
    
    (train_loss_history, train_rec_loss_history, train_kl_div_b_history, train_kl_div_a_history, train_kl_div_d_history,
     val_loss_history, val_rec_loss_history, val_kl_div_b_history, val_kl_div_a_history, val_kl_div_d_history) = [],[],[],[],[],[],[],[],[],[]
    
    epoch_best=0
    print("Training starts...")
    epoch_iterator = tqdm(range(nepochs))
    for epoch in epoch_iterator:

        model.train()
        loss_epoch_s, rec_loss_epoch_s, kl_div_b_epoch_s, kl_div_a_epoch_s, kl_div_d_epoch_s = 0, 0, 0, 0, 0
        for batch in train_dataloader:            
            if type(batch) == list: 
                batch = batch[0]                     
            
            l_s, rl_s, klb_s, kla_s, kld_s  = fit_step_wb_blank_prior(model,batch.to(model.device), variational_beta, variational_alpha, variational_delta)
            loss_epoch_s += l_s
            rec_loss_epoch_s += rl_s
            kl_div_b_epoch_s += klb_s
            kl_div_a_epoch_s += kla_s
            kl_div_d_epoch_s += kld_s

        model.eval()

        val_loss_epoch, val_rec_loss_epoch, val_kl_div_b_epoch, val_kl_div_a_epoch, val_kl_div_d_epoch = 0, 0, 0, 0, 0
        
        for val_batch in validation_dataloader:
            if type(val_batch) == list: 
                val_batch = val_batch[0]

            with torch.no_grad():   
                val_batch = val_batch.to(model.device)
                val_batch_recon_sample, val_mu_sample, val_logvar_sample= model(val_batch)

                vrl, vklb, vkla, vkld = model.vae_loss_wb_blank_prior(val_batch, val_batch_recon_sample, val_mu_sample, val_logvar_sample)
                vl = vae_wb_blank_prior_cost_function(vrl, vklb, vkla, vkld, variational_beta, variational_alpha, variational_delta)

                val_loss_epoch += vl.item()
                val_rec_loss_epoch += vrl.item()
                val_kl_div_b_epoch += vklb.item()
                val_kl_div_a_epoch += vkla.item()
                val_kl_div_d_epoch += vkld.item()

        val_loss_history.append(val_loss_epoch/model.nval)
        val_rec_loss_history.append(val_rec_loss_epoch/model.nval)
        val_kl_div_b_history.append(val_kl_div_b_epoch/model.nval)
        val_kl_div_a_history.append(val_kl_div_a_epoch/model.nval)
        val_kl_div_d_history.append(val_kl_div_d_epoch/model.nval)

        train_loss_history.append(loss_epoch_s/model.ntrain)
        train_rec_loss_history.append(rec_loss_epoch_s/model.ntrain)  
        train_kl_div_b_history.append(kl_div_b_epoch_s/model.ntrain)
        train_kl_div_a_history.append(kl_div_a_epoch_s/model.ntrain)
        train_kl_div_d_history.append(kl_div_d_epoch_s/model.ntrain)
                        
        if ((epoch == 0)or((val_loss_epoch/model.nval) < val_loss_history[epoch_best])):
            epoch_best = epoch
            torch.save(model.state_dict(), saving_path + f"bestnet.pth")
            torch.save(epoch_best, saving_path + f"bestepoch.pth") 
            torch.save(torch.tensor([train_loss_history, train_rec_loss_history, train_kl_div_b_history, train_kl_div_a_history, train_kl_div_d_history,val_loss_history, val_rec_loss_history, val_kl_div_b_history, val_kl_div_a_history, val_kl_div_d_history],requires_grad=False).T,saving_path +f"./bestloss.pth")     

        if (epoch+1)%50==0: #checkpoints every 50 epohcs
            torch.save(model.state_dict(), saving_path +f"./checkpoint_epoch_{(epoch+1):04d}.pth")
        
        epoch_iterator.set_postfix({'best-epoch': epoch_best,
                                    't-loss': train_loss_history[epoch_best],
                                    'v-loss': val_loss_history[epoch_best],
                                    't-rec-loss': train_rec_loss_history[epoch_best],
                                    'v-rec-loss': val_rec_loss_history[epoch_best],
                                    't-kl-beta': train_kl_div_b_history[epoch_best],
                                    'v-kl-beta': val_kl_div_b_history[epoch_best],
                                    't-kl-alpha': train_kl_div_a_history[epoch_best],
                                    'v-kl-alpha': val_kl_div_a_history[epoch_best],
                                    't-kl-delta': train_kl_div_d_history[epoch_best],
                                    'v-kl-delta': val_kl_div_d_history[epoch_best]}
                                    )
        
    torch.save(torch.tensor([train_loss_history, train_rec_loss_history, train_kl_div_b_history, train_kl_div_a_history, train_kl_div_d_history,val_loss_history, val_rec_loss_history, val_kl_div_b_history, val_kl_div_a_history, val_kl_div_d_history],requires_grad=False).T,saving_path + f"finalloss.pth")
    print('...training ended.')
    return


def replace_point_by_underscore(string):
    string = string.replace('.','_')
    return string 

def modify_contrast_x(x_orig,new_contrast):
    orig_contrast = x_orig.std(dim=(1),keepdim=True)
    x_new = x_orig*new_contrast/orig_contrast
    
    return x_new