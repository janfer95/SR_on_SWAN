# SR_on_SWAN
[![DOI](https://zenodo.org/badge/452716476.svg)](https://zenodo.org/badge/latestdoi/452716476)

Code repository for the workflow presented in the article "A neural network super-resolution approach for the reconstruction of coastal sea states"
(J. Kuehn, S. Abadie, V. Roeber, B. Liquet, 2022). 

It contains the SWAN script files that were used to produce the low- and high-resolution results. The pre-processing of the data and the training
of the neural network are done with Python and are saved as Jupyter Notebooks. 

Some example reconstructions distributed over the whole year of test data are given along with time series for three different locations. The jupyter notebooks are commented and should be self-explanatory. 

The Run\_Training.py is a simple python script that runs Train\_Model.py with the chosen parameters. The four different network architectures / 
modelling approaches can be found in Models.py. 

Please contact jannik.kuhn@univ-pau.fr for queries and further material.

![HR SWAN Reconstruction by a Super-Resolution Neural Network](./ExampleImages/ExampleReconstruction.png)
