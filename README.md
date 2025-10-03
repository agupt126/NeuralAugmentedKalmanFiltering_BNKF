# EN705742 Advanced Applied Machine Learning Final_Project

## Author: Akhil Gupta | JHUEP Fall 2024 | agupt126@jh.edu

### Description

This repository serves as a code base for a proposed Bayesian Neural Kalman Filter (BNKF) implementation and performance comparison using the Stonesoup Library.


### Setup
1) Download/clone this repository.
2) Install all required Python packages within `requirements.txt` in a conda environment. Please ensure you have at least Python 3.10 installed.
3) Download data from [HuggingFace Repository](https://huggingface.co/datasets/riotu-lab/Synthetic-UAV-Flight-Trajectories) and store .csv files within a folder named `dataset/Synthetic-UAV-Flight-Trajectories`. You should have a .csv for each trajectory. Please reach out to Akhil if you are having difficulty with data retrieval in the proper format.
4) Customize path inputs as needed.

   Note: There are pre-trained torch weights available for import for experimentation and to make the notebooks run faster. 

### Navigating Repository
In terms of code, the repository is divided into three major modules.
* `helper/`: Contains utility functions. If you want to customize data read-in paths and training features see `helper/training_preprocess.py`.
* `models/`: Contains model code and weights.
* `notebooks/`: Contains main notebooks for experimentation 

### Help?
Please reach out to owner of repository.

### Resources
* Dataset: https://huggingface.co/datasets/riotu-lab/Synthetic-UAV-Flight-Trajectories
* BayesLinear Repo: https://github.com/Harry24k/bayesian-neural-network-pytorch/tree/master
* EKF Source: https://automaticaddison.com
