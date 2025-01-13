## Importing libraries
import sys, os
import numpy as np
import pandas as pd

import pickle as pkl
import scipy.stats as stats

import torch
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, FashionMNIST


import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.colors import LinearSegmentedColormap

def custom_colormap_blue():
        colors_blue = ['#BBCCEE', '#4477AA', '#222255']
        cm_blue = LinearSegmentedColormap.from_list(
                "Custom_blue", colors_blue, N=255)
        cm_blue.set_bad('#FFFFFF')
        #plt.register_cmap(cmap=cm_blue)
        return cm_blue
def custom_colormap_red():
        colors_red = ['#FFCCCC', '#EE6677', '#663333']
        cm_red = LinearSegmentedColormap.from_list(
                "Custom_red", colors_red, N=255)
        cm_red.set_bad('#FFFFFF')
        #plt.register_cmap(cmap=cm_red)
        return cm_red
    
## Analysis functions
def vec_transform(img):
    return img.reshape(-1)

def zero_mean(img):
    img = img - img.mean()
    return img

def data_posteriors(model, dataloader, latent_dim, with_z=False,with_free_z=False):
    model.eval()
    
    model_data_ymu = np.zeros((len(dataloader.dataset), latent_dim), dtype=float)
    model_data_yvar = np.zeros((len(dataloader.dataset), latent_dim), dtype=float)
    model_data_ztheta = np.zeros((len(dataloader.dataset),1), dtype=float)
    model_data_zmu = np.zeros((len(dataloader.dataset),1), dtype=float)
    model_data_zvar = np.zeros((len(dataloader.dataset),1), dtype=float)
    with torch.no_grad():
        done = 0
        for k, (x, y) in enumerate(dataloader):    
            
            n_batch = x.shape[0]
            if with_z:
                _,latent_mus, latent_logvar, z_logtheta = model(x.to(model.device),only_mu=True,contrast_mu=True)
                model_data_ztheta[done:done+n_batch] = z_logtheta.exp().detach().cpu().numpy()
            elif with_free_z:
                _,latent_mus, latent_logvar, z_logtheta = model(x.to(model.device),only_mu=True,only_zmu=True)
                model_data_ztheta[done:done+n_batch] = z_logtheta.exp().detach().cpu().numpy()
                
            else:
                _,latent_mus, latent_logvar = model(x.to(model.device),only_mu=True)
                
            model_data_yvar[done:done+n_batch, :] = latent_logvar.exp().detach().cpu().numpy()
            model_data_ymu[done:done+n_batch, :] = latent_mus.detach().cpu().numpy()
            
            done += n_batch
    if with_z or with_free_z:
        k_param = model.z_k
    else:
        k_param = 1
    model_data_zmu = (k_param)*model_data_ztheta
    model_data_zvar = (k_param)*model_data_ztheta**2
    return model_data_ymu, model_data_yvar, model_data_zmu, model_data_zvar
    
def average_posterior(model_data_ymu, model_data_yvar):
    av_post_mu  = model_data_ymu.mean(axis=0)
    av_post_cov = np.diag(model_data_yvar.mean(axis=0)) + np.cov(model_data_ymu.T)
    
    return av_post_mu, av_post_cov

def individual_posterior(model, img, with_z=False, with_free_z=False):
    model.eval()
    img_ztheta = 0
    with torch.no_grad():
        if with_z:
            _, img_latent_mu, img_latent_logvar, img_z_logtheta  = model(torch.unsqueeze(img.to(model.device),dim=0),only_mu=True, contrast_mu=True)
            img_ztheta = img_z_logtheta.exp().detach().cpu().numpy()
        elif with_free_z:
            _, img_latent_mu, img_latent_logvar, img_z_logtheta  = model(torch.unsqueeze(img.to(model.device),dim=0),only_mu=True, only_zmu=True)
            img_ztheta = img_z_logtheta.exp().detach().cpu().numpy()
            
        else:
            _, img_latent_mu, img_latent_logvar = model(torch.unsqueeze(img.to(model.device),dim=0),only_mu=True)
        
        img_yvar = img_latent_logvar.exp().detach().cpu().numpy().squeeze()
        img_ymu = img_latent_mu.detach().cpu().numpy().squeeze()
        if with_z or with_free_z:
            k_param = model.z_k
        else:
            k_param = 1
        img_zmu = (k_param)*img_ztheta
        img_zvar = (k_param)*img_ztheta**2
    
    return img_ymu, img_yvar, img_zmu, img_zvar

def spike_triggered_average(model, dataloader, imsize, latent_dim, with_z=False, with_free_z=False, X=None, Y=None):
    
    if (type(X) is not type(None)) and (type(Y) is not type(None)):
        return np.matmul(X.T,Y)
    
    model.eval()
    
    model_data_F = np.zeros((imsize, latent_dim), dtype=float)
    
    with torch.no_grad():
        done = 0
        for k, (x, y) in enumerate(dataloader):    
            
            if with_z:
                latent_mus, _, _ = model.encoder(x.to(model.device))
            elif with_free_z:
                latent_mus, _, _ = model.encoder(x.to(model.device))
                
            else:
                latent_mus, _ = model.encoder(x.to(model.device))
                
            ymu = latent_mus.detach().cpu()
            
            model_data_F += torch.matmul(x.T,ymu).numpy()
            
            
    model_data_F/=len(dataloader.dataset)
    
    return model_data_F

def latent_dim_contribution(model, dataloader, latent_dim, with_z=False, with_free_z=False):
    model.eval()

    model_data_ymu = np.zeros((len(dataloader.dataset), latent_dim), dtype=float)
    model_data_zmu = np.zeros((len(dataloader.dataset),1), dtype=float)
    if with_z or with_free_z:
        k_param = model.z_k
    with torch.no_grad():
        done = 0
        for k, (x, y) in enumerate(dataloader):    
            
            n_batch = x.shape[0]
            if with_z or with_free_z:
                latent_mus, _, z_logtheta = model.encoder(x.to(model.device))
                model_data_zmu[done:done+n_batch] = k_param*z_logtheta.exp().detach().cpu().numpy()
            else:
                latent_mus, _ = model.encoder(x.to(model.device))
                
            model_data_ymu[done:done+n_batch, :] = latent_mus.detach().cpu().numpy()
            done += n_batch

    mean_ymu = np.mean(model_data_ymu,axis=0)
    mean_rec_sqr_diff = np.zeros((latent_dim), dtype=float)
    with torch.no_grad():
        for j in range(latent_dim):
            print(j)
            model_data_ymu_j = model_data_ymu.copy()
            model_data_ymu_j[:,j] = mean_ymu[j]
            done = 0
            for k, (x, y) in enumerate(dataloader):
                
                n_batch = x.shape[0]
                if with_z or with_free_z:
                    x_rec = model.decoder(torch.tensor(model_data_ymu[done:done+n_batch, :]).float().to(model.device),torch.tensor(model_data_zmu[done:done+n_batch]).float().to(model.device))
                    x_rec_j = model.decoder(torch.tensor(model_data_ymu_j[done:done+n_batch, :]).float().to(model.device),torch.tensor(model_data_zmu[done:done+n_batch]).float().to(model.device))
                    
                else:
                    x_rec = model.decoder(torch.tensor(model_data_ymu[done:done+n_batch, :]).float().to(model.device))
                    x_rec_j = model.decoder(torch.tensor(model_data_ymu_j[done:done+n_batch, :]).float().to(model.device))
                mean_rec_sqr_diff[j] += torch.sum((x_rec-x_rec_j)**2).detach().cpu().numpy()
                done += n_batch
    mean_rec_sqr_diff /= len(dataloader.dataset)
    return mean_rec_sqr_diff

def SignalMean_SignalVar_NoiseVar(latent_ymu,latent_yvar,latent_z,interest_idxs,with_z=False, with_free_z=False):
    
    signal_mean = np.linalg.norm(latent_ymu[:,interest_idxs],axis=1)
    noise_var = latent_yvar[:,interest_idxs].mean(axis=1)
    contrast_values = latent_z[:]
    #contrast_rounds = np.linspace(np.max(contrast_values)/100,np.max(contrast_values)*(1-1/100),100)
    if with_z:
        with torch.no_grad():
            #contrast_rounds = model_eavae.infer_contrast(torch.tensor((np.linspace(0,1,101)/(6*nat_train_pixs_std)).astype(np.float32),requires_grad=False).to(device),c_mu=True).squeeze(1).cpu().numpy()
            contrast_rounds = np.linspace(np.min(contrast_values),np.max(contrast_values),101)
    elif with_free_z:
        with torch.no_grad():
            contrast_rounds = np.linspace(np.min(contrast_values),np.max(contrast_values),101)
    else:
        #contrast_rounds = np.linspace(0,1,101)
        contrast_rounds = np.linspace(np.min(contrast_values),np.max(contrast_values),101)
        
    rounded_contrast = contrast_rounds[np.argmin(abs(np.subtract.outer(contrast_values, contrast_rounds)), axis=1)]
    
    binned_signal_mean = np.zeros_like(contrast_rounds)
    binned_signal_var = np.zeros_like(contrast_rounds)
    binned_noise_var = np.zeros_like(contrast_rounds)
    for k in range(len(contrast_rounds)):
        binned_signal_mean[k] = signal_mean[rounded_contrast==contrast_rounds[k]].mean()
        binned_signal_var[k] = latent_ymu[:,interest_idxs][rounded_contrast==contrast_rounds[k]].var(axis=0).mean()
        binned_noise_var[k] = noise_var[rounded_contrast==contrast_rounds[k]].mean()
    
    binned_signal_mean_error = np.zeros_like(contrast_rounds)
    binned_signal_var_error = np.zeros_like(contrast_rounds)
    binned_noise_var_error = np.zeros_like(contrast_rounds)
    M = len(interest_idxs)
    N = np.array([sum(rounded_contrast==contrast_rounds[k]) for k in range(len(contrast_rounds))])
    for k in range(len(contrast_rounds)):
        binned_signal_mean_error[k] = signal_mean[rounded_contrast==contrast_rounds[k]].std()/np.sqrt(N[k])
        binned_signal_var_error[k] = binned_signal_var[k]*np.sqrt(2/(N[k]-1))/np.sqrt(M)
        binned_noise_var_error[k] = latent_yvar[rounded_contrast==contrast_rounds[k]][:,interest_idxs].std()/np.sqrt(M*N[k])
    
    return binned_signal_mean, binned_signal_var, binned_noise_var, binned_signal_mean_error, binned_signal_var_error, binned_noise_var_error, contrast_rounds

def plot_losses(losses, labels):
    fig, ax = plt.subplots(1, int(losses.shape[1]/2), figsize=(losses.shape[1]*2,3))
    for k in range(int(losses.shape[1]/2)):
        ax[k].plot(losses[:,k],label='train')
        ax[k].plot(losses[:,k+int(losses.shape[1]/2)],label='val (sample)')
        ax[k].set_xlabel('epoch')
        ax[k].ticklabel_format(axis='y',style='sci',scilimits=(0,0))
        ax[k].set_ylabel(labels[k])
        ax[k].set_yscale('log')

    plt.tight_layout()
    plt.show()