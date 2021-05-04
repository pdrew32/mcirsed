# Installation

For the best results it is recommended that you run the MCIRSED package in a new conda environment unless you want to tinker with your Python package versions. Below are the steps I followed to run MCIRSED on a fresh conda environment in May 2021.

1) Download MCIRSED to your working directory
2) Install anaconda if not done already (https://docs.anaconda.com/anaconda/install/#installation)
3) Create a new conda environment with python 3.7
4) Open a terminal in the new conda environment
5) Update anaconda to the stable build from November 2020: ```conda install anaconda=2020.11```\
(https://docs.anaconda.com/anaconda/reference/release-notes/#anaconda-2020-11-nov-19-2020). \
Note: newer stable builds may work in the future but as of the time of writing this is the latest build.
5) Install pymc3 version 3.8: ```conda install -c conda-forge pymc3=3.8```\
Newer versions may be compatible in the future but for now this is the latest stable version with the anaconda stable build.
6) If using Windows, install g++ compiler using: ```conda install m2w64-toolchain``` 
7) Install libpython: ```conda install libpython```
8) Downgrade arviz to version 0.11.1: ```conda install arviz=0.11.1```
9) Install corner.py for corner plots: ```pip install corner```

# Verifying Installation

1) run example_data.py to generate data for two galaxies
2) run example_fit_mcirsed.py. If the code finishes running and plots of the SED corner plots show up the installation was successful.
