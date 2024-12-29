![LOGO]()

Reproducible material for Multichannel wavefield reconstruction of land seismic data with stencil-based spatial gradients -
Khatami MI.


## Project structure
This repository is organized as follows:

* :open_file_folder: **data**: folder containing the instruction on how to retrieve the data
* :open_file_folder: **landmc**: a set of package to do multichannel wavefield reconstruction of land seismic data
* :open_file_folder: **notebooks**: set of jupyter notebooks reproducing the experiments in the paper (see below for more details);

## Notebooks
The following notebooks are provided:

- :orange_book: ``1. Data Preparation-Synthetic Data.ipynb``: notebook performing the preprocessing of the synthetic data;
- :orange_book: ``2. Data Preparation-Field Data.ipynb``: notebook performing the preprocessing of the field data;
- :orange_book: ``3. LandMCReconstruction-Synthetic Data.ipynb``: notebook performing wavefield reconstruction of synthetic data;
- :orange_book: ``4. LandMCReconstruction-Field Data.ipynb``: notebook performing wavefield reconstruction of field data;


## Getting started :space_invader: :robot:
To ensure reproducibility of the results, we suggest using the `environment.yml` file when creating an environment.

Simply run:
```
./install_env.sh
```
It will take some time, if at the end you see the word `Done!` on your terminal you are ready to go. After that you can simply install your package:
```
pip install .
```
or in developer mode:
```
pip install -e .
```

Remember to always activate the environment by typing:
```
conda activate my_env
```

Finally, to run tests simply type:
```
pytest
```
