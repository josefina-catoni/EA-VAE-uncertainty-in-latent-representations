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
  
  
class VEncoder(nn.Module):
    def __init__(self,latent_size, capacity):
        super(VEncoder, self).__init__()
        self.latent_size = latent_size
        self.c = capacity
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=self.c, kernel_size=4, stride=2, padding=1) # out: c x 16*16
        self.conv2 = nn.Conv2d(in_channels=self.c, out_channels=self.c*2, kernel_size=4, stride=2, padding=1) # out: c*2 x 8*8
        self.conv3 = nn.Conv2d(in_channels=self.c*2, out_channels=self.c*2*2, kernel_size=4, stride=2, padding=1) # out: c*2*2 x 4*4
        self.fc_mu = nn.Linear(in_features=self.c*2*2*4*4, out_features=self.latent_size)
        self.fc_logvar = nn.Linear(in_features=self.c*2*2*4*4, out_features=self.latent_size)            
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1) # flatten batch of multi-channel feature maps to a batch of feature vectors
        
        x_mu = self.fc_mu(x)
        x_logvar = self.fc_logvar(x)
        return x_mu, x_logvar

class VDecoder_BCE(nn.Module):
    def __init__(self,latent_size, capacity):
        super(VDecoder_BCE, self).__init__()
        self.latent_size=latent_size
        self.c = capacity
        self.fc = nn.Linear(in_features=latent_size, out_features=self.c*2*2*4*4)
        self.conv3 = nn.ConvTranspose2d(in_channels=self.c*2*2, out_channels=self.c*2, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.ConvTranspose2d(in_channels=self.c*2, out_channels=self.c, kernel_size=4, stride=2, padding=1)
        self.conv1 = nn.ConvTranspose2d(in_channels=self.c, out_channels=1, kernel_size=4, stride=2, padding=1)
  
    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), self.c*2*2, 4, 4) # unflatten batch of feature vectors to a batch of multi-channel feature maps
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv2(x))
        x = torch.sigmoid(self.conv1(x)) # last layer before output is sigmoid, since we are using BCE as reconstruction loss
        #x = self.conv1(x) # Without sigmoid, associated to MSE reconstruction loss
        return x
class VDecoder_MSE(nn.Module):
    def __init__(self,latent_size, capacity):
        super(VDecoder_MSE, self).__init__()
        self.latent_size=latent_size
        self.c = capacity
        self.fc = nn.Linear(in_features=latent_size, out_features=self.c*2*2*4*4)
        self.conv3 = nn.ConvTranspose2d(in_channels=self.c*2*2, out_channels=self.c*2, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.ConvTranspose2d(in_channels=self.c*2, out_channels=self.c, kernel_size=4, stride=2, padding=1)
        self.conv1 = nn.ConvTranspose2d(in_channels=self.c, out_channels=1, kernel_size=4, stride=2, padding=1)
  
    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), self.c*2*2, 4, 4) # unflatten batch of feature vectors to a batch of multi-channel feature maps
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv2(x))
        x = self.conv1(x) # Without sigmoid, associated to MSE reconstruction loss
        return x
class VDecoder_softplus(nn.Module):
    def __init__(self,latent_size, capacity):
        super(VDecoder_softplus, self).__init__()
        self.latent_size=latent_size
        self.c = capacity
        self.fc = nn.Linear(in_features=latent_size, out_features=self.c*2*2*4*4)
        self.conv3 = nn.ConvTranspose2d(in_channels=self.c*2*2, out_channels=self.c*2, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.ConvTranspose2d(in_channels=self.c*2, out_channels=self.c, kernel_size=4, stride=2, padding=1)
        self.conv1 = nn.ConvTranspose2d(in_channels=self.c, out_channels=1, kernel_size=4, stride=2, padding=1)
  
    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), self.c*2*2, 4, 4) # unflatten batch of feature vectors to a batch of multi-channel feature maps
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv2(x))
        x = F.softplus(self.conv1(x)) # With softplus
        return x
    
class VariationalAutoencoder(nn.Module):
    def __init__(self,ntrain,nval,input_size, latent_size,capacity,l_rate,rec_loss_method, device):
        super(VariationalAutoencoder, self).__init__()
        self.ntrain=ntrain
        self.nval=nval
        self.latent_size=latent_size
        self.input_size = input_size
        self.capacity=capacity
        self.rec_loss_method = rec_loss_method
        self.encoder = VEncoder(self.latent_size, self.capacity)
        
        if self.rec_loss_method == 'BCE':
            self.decoder = VDecoder_BCE(self.latent_size, self.capacity)
            self.vae_loss = self.vae_loss_BCE
            
        elif self.rec_loss_method == 'MSE':
            self.decoder = VDecoder_MSE(self.latent_size, self.capacity)
            self.vae_loss = self.vae_loss_MSE
        elif self.rec_loss_method == 'MSE+softplus':
            self.decoder = VDecoder_softplus(self.latent_size, self.capacity)
            self.vae_loss = self.vae_loss_MSE            
        else:
            raise ValueError("rec_loss_method must be 'BCE' or 'MSE'")
        
        self.optimizer = torch.optim.Adam(params=self.parameters(), lr=l_rate, weight_decay=1e-5)
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
        
    def vae_loss_BCE(self, x, recon_x, mu, logvar):
        recon_loss = F.binary_cross_entropy(recon_x.view(-1, 1024), x.view(-1, 1024), reduction='sum')/self.input_size
        kl_div = self.kl_div_unit_prior(mu, logvar)/self.input_size
  
        return recon_loss, kl_div
    
    def vae_loss_MSE(self, x, recon_x, mu, logvar):
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')/self.input_size
        kl_div = self.kl_div_unit_prior(mu, logvar)/self.input_size
  
        return recon_loss, kl_div

class z_VEncoder(nn.Module):
    def __init__(self,latent_size, capacity):
        super(z_VEncoder, self).__init__()
        self.latent_size = latent_size
        self.c = capacity
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=self.c, kernel_size=4, stride=2, padding=1) # out: c x 16*16
        self.conv2 = nn.Conv2d(in_channels=self.c, out_channels=self.c*2, kernel_size=4, stride=2, padding=1) # out: c*2 x 8*8
        self.conv3 = nn.Conv2d(in_channels=self.c*2, out_channels=self.c*2*2, kernel_size=4, stride=2, padding=1) # out: c*2*2 x 4*4
        self.fc_mu = nn.Linear(in_features=self.c*2*2*4*4, out_features=self.latent_size)
        self.fc_logvar = nn.Linear(in_features=self.c*2*2*4*4, out_features=self.latent_size)     
        self.fc_intz = nn.Linear(in_features=self.c*16*16, out_features=20)
        self.fc_zmu = nn.Linear(in_features=20, out_features=1)
        self.fc_zlogvar = nn.Linear(in_features=20, out_features=1)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        intz = F.softplus(self.fc_intz(x.view(x.size(0), -1)))
        
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1) # flatten batch of multi-channel feature maps to a batch of feature vectors
        
        x_mu = self.fc_mu(x)
        x_logvar = self.fc_logvar(x)
        
        z_mu = self.fc_zmu(intz)
        z_logvar = self.fc_zlogvar(intz)
        
        #z_mu = z_mu.view(x.size(0), 1, 1, 1)
        #z_logvar = z_logvar.view(x.size(0), 1, 1, 1)
        
        return x_mu, x_logvar, z_mu, z_logvar
    
class z_VDecoder_MSE(nn.Module):
    def __init__(self,latent_size, capacity):
        super(z_VDecoder_MSE, self).__init__()
        self.latent_size=latent_size
        self.c = capacity
        self.fc = nn.Linear(in_features=latent_size, out_features=self.c*2*2*4*4)
        self.conv3 = nn.ConvTranspose2d(in_channels=self.c*2*2, out_channels=self.c*2, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.ConvTranspose2d(in_channels=self.c*2, out_channels=self.c, kernel_size=4, stride=2, padding=1)
        self.conv1 = nn.ConvTranspose2d(in_channels=self.c, out_channels=1, kernel_size=4, stride=2, padding=1)
  
    def forward(self, y, z):
        x = self.fc(y)
        x = x.view(x.size(0), self.c*2*2, 4, 4) # unflatten batch of feature vectors to a batch of multi-channel feature maps
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv2(x))
        x = self.conv1(x)
        x = z.exp().view(x.size(0),1,1,1)*x
        return x
    
class z_VDecoder_softplus(nn.Module):
    def __init__(self,latent_size, capacity):
        super(z_VDecoder_softplus, self).__init__()
        self.latent_size=latent_size
        self.c = capacity
        self.fc = nn.Linear(in_features=latent_size, out_features=self.c*2*2*4*4)
        self.conv3 = nn.ConvTranspose2d(in_channels=self.c*2*2, out_channels=self.c*2, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.ConvTranspose2d(in_channels=self.c*2, out_channels=self.c, kernel_size=4, stride=2, padding=1)
        self.conv1 = nn.ConvTranspose2d(in_channels=self.c, out_channels=1, kernel_size=4, stride=2, padding=1)
  
    def forward(self, y, z):
        x = self.fc(y)
        x = x.view(x.size(0), self.c*2*2, 4, 4) # unflatten batch of feature vectors to a batch of multi-channel feature maps
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv2(x))
        x = self.conv1(x)
        x = F.softplus(z.exp().view(x.size(0),1,1,1)*x) # last layer before output is softplus
        return x
    
class z_VDecoder_BCE(nn.Module):
    def __init__(self,latent_size, capacity):
        super(z_VDecoder_BCE, self).__init__()
        self.latent_size=latent_size
        self.c = capacity
        self.fc = nn.Linear(in_features=latent_size, out_features=self.c*2*2*4*4)
        self.conv3 = nn.ConvTranspose2d(in_channels=self.c*2*2, out_channels=self.c*2, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.ConvTranspose2d(in_channels=self.c*2, out_channels=self.c, kernel_size=4, stride=2, padding=1)
        self.conv1 = nn.ConvTranspose2d(in_channels=self.c, out_channels=1, kernel_size=4, stride=2, padding=1)
  
    def forward(self, y, z):
        x = self.fc(y)
        x = x.view(x.size(0), self.c*2*2, 4, 4) # unflatten batch of feature vectors to a batch of multi-channel feature maps
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv2(x))
        x = self.conv1(x)
        x = torch.sigmoid(z.exp().view(x.size(0),1,1,1)*x) # last layer before output is sigmoid, since we are using BCE as reconstruction loss
        
        return x
    
class z_VariationalAutoencoder(nn.Module):
    def __init__(self,ntrain,nval,input_size,latent_size,capacity,l_rate,rec_loss_method, device):
        super(z_VariationalAutoencoder, self).__init__()
        self.ntrain=ntrain
        self.nval=nval
        self.input_size = input_size
        self.latent_size=latent_size
        self.capacity=capacity
        self.rec_loss_method = rec_loss_method
        
        self.encoder = z_VEncoder(self.latent_size, self.capacity)
        
        if self.rec_loss_method == 'BCE':
            self.decoder = z_VDecoder_BCE(self.latent_size, self.capacity)
            self.vae_loss = self.vae_loss_BCE
            
        elif self.rec_loss_method == 'MSE':
            self.decoder = z_VDecoder_MSE(self.latent_size, self.capacity)
            self.vae_loss = self.vae_loss_MSE
        elif self.rec_loss_method == 'MSE+softplus':
            self.decoder = z_VDecoder_softplus(self.latent_size, self.capacity)
            self.vae_loss = self.vae_loss_MSE
        else:
            raise ValueError("rec_loss_method must be 'BCE' or 'MSE'")
        self.optimizer = torch.optim.Adam(params=self.parameters(), lr=l_rate, weight_decay=1e-5)
        self.device = device

    def forward(self, x, only_mu=False, only_zmu=False):
        latent_mu, latent_logvar, zlatent_mu, zlatent_logvar = self.encoder(x)
        latent = self.latent_sample(latent_mu, latent_logvar, only_mu)
        zlatent = self.zlatent_sample(zlatent_mu, zlatent_logvar, only_zmu)
        x_recon = self.decoder(latent,zlatent)
        return x_recon, latent_mu, latent_logvar, zlatent_mu, zlatent_logvar

    def latent_sample(self, mu, logvar, only_mu=False):
        if only_mu:
            return mu
        else:
             # the reparameterization trick
            std = logvar.mul(0.5).exp_()
            eps = torch.empty_like(std).normal_()
            return eps.mul(std).add_(mu)               
   
    def zlatent_sample(self, zmu, zlogvar, only_zmu=False):
        if only_zmu:
            return zmu
        else:
             # the reparameterization trick
            std = zlogvar.mul(0.5).exp_()
            eps = torch.empty_like(std).normal_()
            return eps.mul(std).add_(zmu)  
        
    def kl_div_unit_prior(self, mu, logvar):
        var_inv_var = logvar.exp()
        log_ratio_vars = -logvar
        weighted_dmu2 = mu.pow(2)

        kldivergence =  0.5 * torch.sum( -1 + log_ratio_vars + weighted_dmu2 + var_inv_var)
        
        return kldivergence
        
    def vae_loss_BCE(self, x, recon_x, mu, logvar, zmu, zlogvar):
        recon_loss = F.binary_cross_entropy(recon_x.view(-1, 1024), x.view(-1, 1024), reduction='sum')/self.input_size
        #recon_loss = F.mse_loss(recon_x, x, reduction='sum')/self.input_size
        kl_div = self.kl_div_unit_prior(mu, logvar)/self.input_size
        kl_div_z = self.kl_div_unit_prior(zmu, zlogvar)/self.input_size
        return recon_loss, kl_div, kl_div_z
    
    def vae_loss_MSE(self, x, recon_x, mu, logvar, zmu, zlogvar):
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')/self.input_size
        kl_div = self.kl_div_unit_prior(mu, logvar)/self.input_size
        kl_div_z = self.kl_div_unit_prior(zmu, zlogvar)/self.input_size
        return recon_loss, kl_div, kl_div_z
    
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

def z_vae_cost_function(recon_loss, kl_div, kl_div_z, beta_y, beta_z):
    loss = recon_loss + beta_y * kl_div + beta_z * kl_div_z
    return loss

def z_fit_step(model, x, beta_y, beta_z):
    x_hat, latent_mu, latent_logvar, zlatent_mu, zlatent_logvar = model(x)
    recon_loss, kl_div, kl_div_z = model.vae_loss(x, x_hat, latent_mu, latent_logvar, zlatent_mu, zlatent_logvar)
    loss = z_vae_cost_function(recon_loss, kl_div, kl_div_z, beta_y, beta_z)
    model.optimizer.zero_grad()
    loss.backward() 
    model.optimizer.step() 

    return loss.item(), recon_loss.item(), kl_div.item(), kl_div_z.item()

def z_train_and_val(model, train_dataloader, validation_dataloader, nepochs, variational_beta_y, variational_beta_z, saving_path):
    
    (train_loss_history, train_rec_loss_history, train_kl_div_history, train_kl_div_z_history,
    val_loss_history, val_rec_loss_history, val_kl_div_history, val_kl_div_z_history) = [],[],[],[],[],[],[],[]
    
    epoch_best=0
    print("Training starts...")
    epoch_iterator = tqdm(range(nepochs))
    for epoch in epoch_iterator:

        model.train()
        loss_epoch_s, rec_loss_epoch_s, kl_div_epoch_s, kl_div_z_epoch_s = 0., 0., 0., 0.
        for batch in train_dataloader:            
            if type(batch) == list: 
                batch = batch[0]                     
            
            l_s, rl_s, kl_s, klz_s  = z_fit_step(model,batch.to(model.device),variational_beta_y, variational_beta_z)
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
                val_batch_recon_sample, val_mu_sample, val_logvar_sample, val_zmu_sample, val_zlogvar_sample = model(val_batch)
                
                vrl, vkl, vklz =  model.vae_loss(val_batch, val_batch_recon_sample, val_mu_sample, val_logvar_sample, val_zmu_sample, val_zlogvar_sample)
                vl = z_vae_cost_function(vrl, vkl, vklz, variational_beta_y, variational_beta_z)
                val_loss_epoch += vl.item()
                val_rec_loss_epoch += vrl.item()
                val_kl_div_epoch += vkl.item()
                val_kl_div_z_epoch += vklz.item()

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

        if (epoch+1)%100==0: #checkpoints every 100 epohcs
            torch.save(model.state_dict(), saving_path +f"./checkpoint_epoch_{(epoch+1):04d}.pth")
            torch.save(torch.tensor([train_loss_history, train_rec_loss_history, train_kl_div_history, train_kl_div_z_history, val_loss_history, val_rec_loss_history, val_kl_div_history, val_kl_div_z_history],requires_grad=False).T,saving_path + f"finalloss.pth")
        
        epoch_iterator.set_postfix({'best-epoch': epoch_best,
                                    't-loss': train_loss_history[epoch_best],
                                    'v-loss': val_loss_history[epoch_best],
                                    't-rec-loss': train_rec_loss_history[epoch_best],
                                    'v-rec-loss': val_rec_loss_history[epoch_best],
                                    't-kly-div': train_kl_div_history[epoch_best],
                                    'v-kly-div': val_kl_div_history[epoch_best],
                                    't-klz-div': train_kl_div_z_history[epoch_best],
                                    'v-klz-div': val_kl_div_z_history[epoch_best]}
                                    )
        
    torch.save(torch.tensor([train_loss_history, train_rec_loss_history, train_kl_div_history, train_kl_div_z_history, val_loss_history, val_rec_loss_history, val_kl_div_history, val_kl_div_z_history],requires_grad=False).T,saving_path + f"finalloss.pth")
    print('...training ended.')
    return


def replace_point_by_underscore(string):
    string = string.replace('.','_')
    return string 

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
    
class MNISTShuffDataset(Dataset):
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
    
    
class cMNISTDataset(Dataset):
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