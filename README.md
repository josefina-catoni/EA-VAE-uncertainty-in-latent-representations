# EA-VAE-uncertainty-in-latent-representations

### 1. **Create and activate the Virtual Environment**
Navigate to the directory where you want to create your virtual environment:

```bash
cd /path/to/your/project
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

### 2. **Install Packages**
After activating the environment, you can

- Install Pytorch https://pytorch.org/

- Install required libraries

```bash
pip install -r requirements.txt
```

---
### 3. **Create Van Hateren Dataset**
- Download from https://zenodo.org/record/6584838#.ZBSxatLMJkg 40x40pix natural images patches file: fakelabeled_natural_commonfiltered_640000_40px.pkl and place it inside ./Datasets/VanHateren/ directory.

- Extract dataset by running ./Datasets/VanHateren/extract_dataset.py file
---