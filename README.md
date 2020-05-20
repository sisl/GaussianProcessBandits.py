# GaussianProcessBandits.py
A framework for Gaussian Process Bandits for automatic hyperparameter tuning

## File stucture
- `GaussianProcessBandits` contains the source code for the package
- `datasets` contains datasets pertaining to experiments.

## Environment
A conda environment for this repo can be set up and uses with:
```
conda create -y --name gpb python==3.7
conda install -y --name gpb -c conda-forge pytorch --file requirements.txt
conda activate gpb
...
conda deactivate
```
