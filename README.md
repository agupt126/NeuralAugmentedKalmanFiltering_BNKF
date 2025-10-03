# EN705742 Advanced Applied Machine Learning Final_Project

## Author: Akhil Gupta | JHUEP Fall 2024 | agupt126@jh.edu

### Description

This repository serves as a codebase for my final project submission of the research topic "Nonlinear and Noisy State Estimation with Extended Kalman Physics-Informed Bayesian Neural Network Filter". All necessary models and comaprison notebooks required to recreate the topic analysis are provided here. We show that the PIBNN is able to outperform its competitors in the nonlinear state estimation context of our modeled UAV problem. 


### Setup
1) Download/clone this repository.
2) Install all required Python packages within `requirements.txt` in a conda environment. Please ensure you have at least Python 3.10 installed.
3) Download data from [HuggingFace Repository](https://huggingface.co/datasets/riotu-lab/Synthetic-UAV-Flight-Trajectories) and store .csv files within a folder named `dataset/Synthetic-UAV-Flight-Trajectories`. You should have a .csv for each trajectory. Please reach out to Akhil if you are having difficulty with data retrieval in the proper format.
4) Customize path inputs if needed within `helper/training_preprocess.py` and run test comparison notebooks within `notebooks/comparisons`.

   Note: There are pre-trained torch weights available for import for experimentation and to make the notebooks run faster. You can train models from scratch by setting appropriate flag however this will take some time.

### Navigating Repository
In terms of code, the repository is divided into three major modules.
* `helper/`: Contains utility functions. If you want to customize data read-in paths and training features see `helper/training_preprocess.py`.
* `models/`: Contains model code and weights.
* `notebooks/`: Contains main notebooks for experimentation.

### Example




### Resources
* See `Project_Paper.pdf` for more information/documentation of project.
* Dataset: https://huggingface.co/datasets/riotu-lab/Synthetic-UAV-Flight-Trajectories
* BayesLinear Repo: https://github.com/Harry24k/bayesian-neural-network-pytorch/tree/master
* EKF Source: https://automaticaddison.com
