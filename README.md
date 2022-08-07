# BCI P300 Speller

Data source: <https://www.kaggle.com/rramele/p300samplingdataset>

Related paper: <https://www.frontiersin.org/articles/10.3389/fncom.2019.00043/full>

Colab file: <https://drive.google.com/file/d/1O8k0LcvALgtEFIesd3hDMDRDHs8VS9Mg/view?usp=sharing>

## Dataset extra info

- Each subject spell the same word/letter, but the flash data is different (since it is random order). However, if you just need
  the flash's simulation id (of row and columns) to test correctness of prediction, flash data from one subject (one file) is ok for everyone.
- Each file is one subject, comprise of 35 trials, but last trial is incompleted so we only use 34.;

## Command

Required Anaconda or Miniconda installed.

Create a new virtual environment and install packages

```python
conda env create -f environment.yml
```

Update environment

```python
conda env update -f environment.yml --prune
```

To run

```python
python main.py
```
