import sys, os
import pickle as pkl
import pandas as pd
import numpy as np

# directory where the Zenodo file is located
current_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_path)
natural40_root= current_dir+'/'

# directories where the images will be stored
natural40_train_dir = natural40_root + 'train_images/'
natural40_test_dir = natural40_root + 'test_images/'

if not os.path.exists(natural40_train_dir):
   os.makedirs(natural40_train_dir)

if not os.path.exists(natural40_test_dir):
   os.makedirs(natural40_test_dir)

natural_images_40 = pd.read_pickle(natural40_root+ 'fakelabeled_natural_commonfiltered_640000_40px.pkl')

with open(natural40_root+'test_labels.pkl','wb') as f:
	pkl.dump(natural_images_40['test_labels'], f)

with open(natural40_root+'train_labels.pkl','wb') as f:
	pkl.dump(natural_images_40['train_labels'], f)

for i in range(len(natural_images_40['test_images'])):
   with open(natural40_test_dir+f'{i}.pkl','wb') as f:
       pkl.dump(natural_images_40['test_images'][i], f)

for i in range(len(natural_images_40['train_images'])):
   with open(natural40_train_dir+f'{i}.pkl','wb') as f:
       pkl.dump(natural_images_40['train_images'][i], f)
       
np.save(natural40_test_dir+'test_images.npy', natural_images_40['test_images'])
np.save(natural40_train_dir+'train_images.npy', natural_images_40['train_images'])