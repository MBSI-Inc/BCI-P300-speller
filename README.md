# BCI P300 Speller

## Pygame section

- game_with_eeg.py
- Character.py
- MockExplore.py

The code is designed to be adjustible so that the the timings, letters and loops can be changed.

The game will run through each row and column `n_cycles_in_epoch` times, after which it will either wait `break_time` seconds before starting a new epoch or wait until the user presses space, depending of if `auto_epoch` is true.

## Command

Required Anaconda or Miniconda installed.

Create a new virtual environment and install packages

`conda env create -f environment.yml`

Update environment

`conda env update -f environment.yml --prune`

Run with recording data from Mentalab Explore device

`python game_with_eeg.py -n [device_name] -f [output_file_name]`

If you don't have access to the device or just want to make change to pygame code:

`python game_with_eeg.py -m true`

To train model with data collected from Mentalab Explore device

`python train_model.py`
