# EA-VAE-uncertainty-in-latent-representations
This repository contains the code that accompanies the manuscript *Uncertainty in latent representations of variational autoencoders optimized for visual tasks*. It allows reproduction of the results presented in the paper.

## Instructions

### 1. Clone the repository:
Before using the repository, please make sure you have **Git LFS** installed to manage large files. You can install Git LFS by:
```bash
sudo apt-get install git-lfs
```

After installing Git LFS, run the following command to initialize it:
```bash
git lfs install
```

Once Git LFS is installed and configured, you can proceed with cloning the repository:
```bash
git clone https://github.com/josefina-catoni/EA-VAE-uncertainty-in-latent-representations.git
```

Git LFS will automatically download the large files during the clone or pull process.

---

### 2. **Create and activate a Virtual Environment**
Navigate to the directory where you want to create your virtual environment:

```bash
cd /path/to/your/project/EA-VAE-uncertainty-in-latent-representations
```

Create a virtual environment with conda (Download and install first):

```bash
conda create --name my_env python=3.11.4
```

- Replace `myenv` with the name you want for the virtual environment.

**Activate the Virtual Environment**

- On Linux:
  ```bash
  conda activate myenv
  ```

You should see the environment name (`myenv`) in your terminal prompt, indicating the virtual environment is active.

---

### 3. **Install Packages**
After activating the environment, you can

- Install Pytorch https://pytorch.org/

- Install required libraries

```bash
pip install -r requirements.txt
```

---
### 4. **Create Van Hateren Dataset**
- Download from https://zenodo.org/record/6584838#.ZBSxatLMJkg 40x40pix natural images patches file: fakelabeled_natural_commonfiltered_640000_40px.pkl and place it inside Datasets/VanHateren/ directory.

- Extract dataset by running Datasets/VanHateren/extract_dataset.py file
---

## Folder Structure

- **Plot_results**: Contains Jupyter notebooks to visualize and recreate the figures from the MNIST and Van Hateren Gamma-Laplace models. You can plots the results shown on the manuscript, or the ones obtained with your own trained model.
  
- **MNIST**: Includes the Jupyter notebooks for training the MNIST model and evaluating its results. Using these notebooks you can first train your own model. Then when evaluating, necessary results for plots will be saved in a folder.
  
- **VanHateren_Gamma-Laplace**: Similar to the MNIST folder, this contains Jupyter notebooks for training and evaluating the VanHateren Gamma-Laplace model.

## Accessing Results

In all Jupyter notebooks, you can set the variable `use_manuscript_results` to either `True` or `False`:
- **`True`**: Will load the checkpoints and results from the manuscript.
- **`False`**: Will use locally trained models or newly trained results.

This allows flexibility for users who may want to either replicate the manuscript's results or train their own models from scratch.
