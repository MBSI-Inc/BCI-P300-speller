# BCI P300 Speller

## Pygame section

- game_with_eeg.py
- Character.py
- MockExplore.py

Reuben's and Lam's code.

The code is designed to be adjustible so that the timings, letters and loops can be changed.

The game will run through each row and column `n_cycles_in_epoch` times, after which it will either wait `break_time` seconds before starting a new epoch or wait until the user presses space, depending of if `auto_epoch` is true.

## Training section

- train_model.py
- utils.py
- analysis.py

Brandon's code.

Data to be trained need to be in `data` folder, which on root folder. Other parameters need to be adjusted in the `setup()` function inside `train_model.py`.

Example data structure:

```python
data/
  |-brandon8hz/
  |-brandon16hz/
    |-brandon16hz_ExG.csv
    |-brandon16hz_Marker.csv
    |-brandon16hz_Meta.csv
    |-brandon16hz_ORN.csv
train_model.py
.gitignore
.vscode/
```

Currently, after training, the model will be saved as `model.joblib`.

In next updates, we will add argument parsers so it's easier to change parameters and load model.

## Command

Required **Anaconda** or **Miniconda** installed.

Create a new virtual environment and install packages

`conda env create -f environment.yml`

Update environment

`conda env update -f environment.yml --prune`

Run with recording data from Mentalab Explore device

`python game_with_eeg.py -n [device_name] -o [output_filename] -m [model_filename]`

Example: `python game_with_eeg.py -n Explore_842F -o data/lam8hz/lam8hz -m model.joblib`

If you don't have access to the device or just want to make change to pygame code:

`python game_with_eeg.py --mock`

To train model with data collected from Mentalab Explore device

`python train_model.py`
