# BGOB_n_test
Microbiome time series analysis - BGOB(Bidirectional GRU-ODE-Bayes) model

## Setting Up the Environment
Before running BGOB with the testing code, you need to create and activate the environment using Anaconda or Miniconda. This installs the package in 'editable' mode, allowing you to make changes to the source code and see the effects immediately.

```bash
conda create --name BGOB python=3
conda activate BGOB
```

## Installation
After setting up the environment, you can install the package.

```
cd BGOB
pip install -e .
```

## Running a Demo
To verify the installation and get a feel for the package, you can run a demonstration in `BGOB_run`. This will execute the main script and run the BGOB model on a sample dataset.

```
cd BGOB_run
python main.py
```
