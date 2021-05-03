# Install Tutorial

For the best results it is recommended that you run the MCIRSED package in a new conda environment unless you want to tinker with the packages yourself. Below are the steps I followed to run MCIRSED on a fresh conda environment in May 2021.

1) Install anaconda if not done already (https://docs.anaconda.com/anaconda/install/#installation)
2) Create a new conda environment with python 3.7
3) Open a terminal in the new conda environment
4) Update anaconda to the stable build from November 2020: ```conda install anaconda=2020.11```\
(https://docs.anaconda.com/anaconda/reference/release-notes/#anaconda-2020-11-nov-19-2020). \
Note: newer stable builds may work in the future but as of the time of writing this is the latest build.
5) Install pymc3 version 3.8: ```conda install -c conda-forge pymc3=3.8```\
Newer versions may be compatible in the future but for now this is the latest stable version with the anaconda stable build.
6) 
