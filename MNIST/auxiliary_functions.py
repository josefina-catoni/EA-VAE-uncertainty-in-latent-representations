
## Importing libraries
import sys, os
import numpy as np
import pandas as pd
import pickle
import pickle as pkl
import scipy.stats as stats
from scipy.stats import norm

import torch
from torch.utils.data import DataLoader, Subset
import torch.utils.data as data
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torchvision.datasets import FashionMNIST
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import scipy
import seaborn as sns

from netcal.metrics import ECE

from matplotlib.colors import LinearSegmentedColormap

#color palettes
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
def remove_mean_transform(img):
    # Apply some manipulation to the image (e.g., invert colors)
    img = img - img.mean()
    return img
def scale_0_1_transform(img):
    min = img.min()
    max = img.max()
    img = (img - min)/(max - min)
    return img

def data_posteriors(model, dataloader, latent_dim, with_z):
    model.eval()
    model_data_mu = np.zeros((len(dataloader.dataset), latent_dim), dtype=float)
    model_data_var = np.zeros((len(dataloader.dataset), latent_dim), dtype=float)
    model_data_zmu = np.zeros((len(dataloader.dataset)), dtype=float)
    model_data_zvar = np.zeros((len(dataloader.dataset)), dtype=float)
    with torch.no_grad():
        done = 0
        for k, (x, y) in enumerate(dataloader):    
            
            n_batch = x.shape[0]
            if with_z:
                _,latent_mus, latent_logvar, z_mu, z_logvar = model(x.to(model.device),only_mu=True, only_zmu=True)
                model_data_zmu[done:done+n_batch] = z_mu.detach().cpu().numpy().squeeze()
                model_data_zvar[done:done+n_batch] = z_logvar.exp().detach().cpu().numpy().squeeze()
            else:
                _,latent_mus, latent_logvar = model(x.to(model.device),only_mu=True)
                
            model_data_var[done:done+n_batch, :] = latent_logvar.exp().detach().cpu().numpy()
            model_data_mu[done:done+n_batch, :] = latent_mus.detach().cpu().numpy()
            
            done += n_batch

    return model_data_mu, model_data_var, model_data_zmu, model_data_zvar
    
def average_posterior(model_data_mu, model_data_var):
    av_post_mu  = model_data_mu.mean(axis=0)
    av_post_var = model_data_var.mean(axis=0) + np.diag(np.cov(model_data_mu.T))
    
    return av_post_mu, av_post_var

def individual_posterior(model, img, with_z):
    model.eval()
    img_zmu, img_zvar = None, None
    with torch.no_grad():
        if with_z:
            _, img_latent_mu, img_latent_logvar, img_z_mu, img_z_logvar  = model(torch.unsqueeze(img.to(model.device),dim=0),only_mu=True, only_zmu=True)
            img_zmu = img_z_mu.detach().cpu().numpy().squeeze()
            img_zvar = img_z_logvar.exp().detach().cpu().numpy().squeeze()
        else:
            _, img_latent_mu, img_latent_logvar = model(torch.unsqueeze(img.to(model.device),dim=0),only_mu=True)
        
        img_var = img_latent_logvar.exp().detach().cpu().numpy().squeeze()
        img_mu = img_latent_mu.detach().cpu().numpy().squeeze()
    
    return img_mu, img_var, img_zmu, img_zvar

def vae_category_interpolation_pixel(model, category_recons, num1, num2, lambda1):
    # lambda1 = 0 => inter_latent = latent_1
    # lambda1 = 1 => inter_latent = latent_2
    with torch.no_grad():
        # interpolated image
        inter_in = (1-lambda1)* category_recons[num1] + lambda1 * category_recons[num2] 
        # reconstructed and latent representation of interpolated image
        inter_out, inter_latent_mu, inter_latent_logvar = model(torch.tensor(inter_in).to(model.device),only_mu=True)

        return inter_in, inter_out.detach().cpu().numpy(), inter_latent_mu.detach().cpu().numpy(), inter_latent_logvar.exp().detach().cpu().numpy()
def vae_categoric_interpolation_line_pixel(model, category_recons, num1, num2, n):
    interpolation_img_in = []
    interpolation_img_out = []
    interpolation_var = []
    interpolation_mu = []
    for lam in np.linspace(0,1,n+1):
        img_in, img_out, mu, var = vae_category_interpolation_pixel(model=model, category_recons=category_recons, num1=num1, num2=num2, lambda1=lam)
        interpolation_var.append(var[0])
        interpolation_mu.append(mu[0])
        
        interpolation_img_in.append(img_in[0])
        interpolation_img_out.append(img_out[0])
    return np.array(interpolation_img_in), np.array(interpolation_img_out), np.array(interpolation_var), np.array(interpolation_mu)

def eavae_category_interpolation_pixel(model, category_recons, num1, num2, lambda1):
    # lambda1 = 0 => inter_latent = latent_1
    # lambda1 = 1 => inter_latent = latent_2
    with torch.no_grad():
        # interpolated image
        inter_in = (1-lambda1)* category_recons[num1] + lambda1 * category_recons[num2] 
        # reconstructed and latent representation of interpolated image
        inter_out, inter_latent_mu, inter_latent_logvar, inter_z_mu, inter_z_logvar = model(torch.tensor(inter_in).to(model.device),only_mu=True, only_zmu=True)

        return inter_in, inter_out.detach().cpu().numpy(), inter_latent_mu.detach().cpu().numpy(), inter_latent_logvar.exp().detach().cpu().numpy(),inter_z_mu.detach().cpu().numpy(), inter_z_logvar.exp().detach().cpu().numpy()
def eavae_categoric_interpolation_line_pixel(model, category_recons, num1, num2, n):
    interpolation_img_in = []
    interpolation_img_out = []
    interpolation_var = []
    interpolation_mu = []
    interpolation_zmu = []
    interpolation_zvar = []
    for lam in np.linspace(0,1,n+1):
        img_in, img_out, mu, var, zmu, zvar  = eavae_category_interpolation_pixel(model=model, category_recons=category_recons, num1=num1, num2=num2, lambda1=lam)
        interpolation_var.append(var[0])
        interpolation_mu.append(mu[0])
        interpolation_zmu.append(zmu[0])
        interpolation_zvar.append(zvar[0])
        interpolation_img_in.append(img_in[0])
        interpolation_img_out.append(img_out[0])
    return np.array(interpolation_img_in), np.array(interpolation_img_out), np.array(interpolation_var), np.array(interpolation_mu), np.array(interpolation_zmu), np.array(interpolation_zvar)

def vae_fusion_posterior_pixel(vae_model, img1, img2, lambda1):
    # lambda1 = 0 => inter_latent = latent_1
    # lambda1 = 1 => inter_latent = latent_2
    with torch.no_grad():
        # interpolated image
        inter_in = (1-lambda1)* img1 + lambda1 * img2 
        # reconstructed and latent representation of interpolated image
        inter_out, inter_latent_mu, inter_latent_logvar = vae_model(inter_in.to(vae_model.device),only_mu=True)
        return inter_in, inter_out.detach().cpu().numpy(), inter_latent_mu.detach().cpu().numpy(), inter_latent_logvar.exp().detach().cpu().numpy()

def vae_fusion_numbers_pixel(vae_model, DataSet, cat1, cat2, N=1, lam = 0.5):
    interpolation_img_in = []
    interpolation_img_out = []
    interpolation_var = []
    interpolation_mu = []
    idxs1 = np.where(DataSet.dataset.targets[DataSet.indices]==cat1)[0][:N]

    if cat1==cat2:
        idxs2 = np.where(DataSet.dataset.targets[DataSet.indices]==cat2)[0][N:2*N]
    else:
        idxs2 = np.where(DataSet.dataset.targets[DataSet.indices]==cat2)[0][:N]    
    
    for n in range(N):
        image1 = DataSet.dataset.__getitem__(DataSet.indices[idxs1[n]])[0].unsqueeze(0)
        image2 = DataSet.dataset.__getitem__(DataSet.indices[idxs2[n]])[0].unsqueeze(0)
        img_in, img_out, mu, var = vae_fusion_posterior_pixel(vae_model, image1, image2, lam)
        interpolation_var.append(var[0])
        interpolation_mu.append(mu[0])
        
        interpolation_img_in.append(img_in[0])
        interpolation_img_out.append(img_out[0])
    
    return np.array(interpolation_img_in), np.array(interpolation_img_out), np.array(interpolation_mu), np.array(interpolation_var) 

def eavae_fusion_posterior_pixel(eavae_model, img1, img2, lambda1):
    # lambda1 = 0 => inter_latent = latent_1
    # lambda1 = 1 => inter_latent = latent_2
    with torch.no_grad():
        # interpolated image
        inter_in = (1-lambda1)* img1 + lambda1 * img2 
        # reconstructed and latent representation of interpolated image
        inter_out, inter_latent_mu, inter_latent_logvar, inter_z_mu, inter_z_logvar = eavae_model(inter_in.to(eavae_model.device),only_mu=True, only_zmu=True)
        return inter_in, inter_out.detach().cpu().numpy(), inter_latent_mu.detach().cpu().numpy(), inter_latent_logvar.exp().detach().cpu().numpy(),inter_z_mu.detach().cpu().numpy(), inter_z_logvar.exp().detach().cpu().numpy()

def eavae_fusion_numbers_pixel(eavae_model, DataSet, cat1, cat2, N=1, lam = 0.5):
    interpolation_img_in = []
    interpolation_img_out = []
    interpolation_var = []
    interpolation_mu = []
    interpolation_zmu = []
    interpolation_zvar = []
    idxs1 = np.where(DataSet.dataset.targets[DataSet.indices]==cat1)[0][:N]

    if cat1==cat2:
        idxs2 = np.where(DataSet.dataset.targets[DataSet.indices]==cat2)[0][N:2*N]
    else:
        idxs2 = np.where(DataSet.dataset.targets[DataSet.indices]==cat2)[0][:N]    
        
    for n in range(N):
        image1 = DataSet.dataset.__getitem__(DataSet.indices[idxs1[n]])[0].unsqueeze(0)
        image2 = DataSet.dataset.__getitem__(DataSet.indices[idxs2[n]])[0].unsqueeze(0)
        img_in, img_out, mu, var, zmu, zvar = eavae_fusion_posterior_pixel(eavae_model, image1, image2, lam)
                
        interpolation_var.append(var[0])
        interpolation_mu.append(mu[0])
        interpolation_zmu.append(zmu[0])
        interpolation_zvar.append(zvar[0])
        interpolation_img_in.append(img_in[0])
        interpolation_img_out.append(img_out[0])
    return np.array(interpolation_img_in), np.array(interpolation_img_out), np.array(interpolation_mu), np.array(interpolation_var), np.array(interpolation_zmu), np.array(interpolation_zvar)
