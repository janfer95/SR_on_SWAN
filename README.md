# SR_on_SWAN
[![DOI](https://zenodo.org/badge/452716476.svg)](https://zenodo.org/badge/latestdoi/452716476)

## General Information

Code repository for the workflow presented in the article "A deep learning super-resolution model to speed up computations of coastal sea states"
(J. Kuehn, S. Abadie, B. Liquet, V. Roeber, 2023). 

It contains the SWAN script files that were used to produce the low- and high-resolution results. The pre-processing of the data and the training
of the neural network are done with Python and are saved as Jupyter Notebooks (see https://jupyter.org/ for more info).

The run_training.py is a simple python script that runs train_model.py or train_surrogate.py with the chosen parameters.  
The keras implementation of the models can be found in models.py.

## Package installation
To run the scripts and the notebooks you need the tensorflow, scikit-learn, matplotlib, and ipykernel packages and python>=3.10. 
For compability of keras / tensorflow with the provided model weights we recommend specifically the version 2.9 for both tensorflow and keras. 

While exact installation instructions are out of the scope of this repository a minimal working solution is the following:

1. Install conda (visit the [conda website](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) for
more information). 

2. Create a conda environment, for example type in a terminal: `conda create -n superres python=3.10`

3. Install the necessary packages with pip (or conda if you like):
`pip install tensorflow==2.9.0 scikit-learn matplotlib ipykernel`


Note that usually this only should install the CPU-version of tensorflow. Check out the tensorflow website for more information
on how to install the GPU version.

## Data
Given the large site of the data set that the neural network was trained on, it is not contained in this repository and has to be downloaded
seperately [here](https://nuage.univ-pau.fr/s/fRMeRxnkj7TyERr). Only the time series extracted at the three locations (see article), along with the pre-processed bathymetry data obtained from the Digital Terrain Model ”MNT bathymétrique de façade Atlantique” provided by the French Service Hydrographique et Océano-graphique de la Marine (SHOM) are included in the repository. Note that certain notebooks / scripts, might not work properly without the downloaded data. 
Additionally, two example snapshots (a good and bad reconstruction of the sea states) are provided for the main plot.

Concerning the model weights, only one example is provided per variable for the super-resolution model.
Due to the large amount of parameters for the surrogate model they have to be downloaded seperately
from the same link as above. 

Also, the Convergence Analysis folder contains only the scripts to give an idea of how it was implemented.

Please contact jannik.kuehn@outlook.de in case of questions or further material.

![HR SWAN Reconstruction by a Super-Resolution Neural Network](./ExampleImages/ExampleReconstruction.png)
