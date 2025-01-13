#Importing libraries
import sys, os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from scipy.stats import multivariate_normal

import seaborn as sns

import pdb

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
    
def gauss2d(label, mu, sigma=[],cov=[], to_plot_contour=False,to_plot_inside=False):

    w, h = 1001, 1001
    if len(sigma)>1:
        std = [sigma[0], sigma[1]]
        x = np.linspace(mu[0] -  2*std[0], mu[0] +  2*std[0], w)
        y = np.linspace(mu[1] -  2*std[1], mu[1] +  2*std[1], h)
    elif len(cov)>1:
        std = [np.sqrt(cov[0,0]), np.sqrt(cov[1,1])]
        x = np.linspace(mu[0] -  2*std[0], mu[0] +  2*std[0], w)
        y = np.linspace(mu[1] -  2*std[1], mu[1] +  2*std[1], h)
    else:
        print('please give sigma or covariance')
    x, y = np.meshgrid(x, y)

    x_ = x.flatten()
    y_ = y.flatten()
    xy = np.vstack((x_, y_)).T
    if len(sigma)>1:
        normal_rv = multivariate_normal(mu, sigma)#, allow_singular=True)
        level = normal_rv.pdf([mu[0]+sigma[0],mu[1]])

    elif len(cov)>1:
        normal_rv = multivariate_normal(mu, cov)#, allow_singular=True)
        eigvals,eigvecs=np.linalg.eig(cov)
        level=normal_rv.pdf(mu+np.sqrt(eigvals[0])*eigvecs[:,0])
    level_mu = normal_rv.pdf(mu)

    z = normal_rv.pdf(xy)

    z = z.reshape(w, h, order='F')


    if to_plot_contour:
        plt.contour(x, y, z.T,levels=[level],colors=[tuple([x*1 for x in sns.color_palette("Set2")[label]]) ],alpha=0.75,linewidths=1,zorder=1)
        #plt.colorbar()
    if to_plot_inside:
        plt.contourf(x, y, z.T,levels=[level,level_mu],colors=[tuple([x*1 for x in sns.color_palette("Set2")[label]]) ],alpha=0.5)
    return z

# Function to create 3D histogram
def create_3d_histogram(ax, predicts, color, title):
    _x = np.arange(10)
    _y = np.linspace(0, 1, 11)
    _xx, _yy = np.meshgrid(_x, _y)
    x, y = _xx.ravel(), _yy.ravel()

    top = predicts.ravel()
    bottom = np.zeros_like(top)
    width = 0.8
    depth = 0.08

    ax.bar3d(x, y, bottom, width, depth, top, color=color)
    ax.set_title(title)
    ax.set_zlim(0, 1)
    ax.set_xlabel('class')
    ax.set_ylabel('mixture')
    ax.set_zlabel('probability')
