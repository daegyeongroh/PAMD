# MICo Residual Test

This code evaluates the fixed-point (Bellman residual) fitting behavior of different latent distance parameterizations using a replay buffer.

## Requirements
- A replay buffer file `buffer.pt`, obtained by running DBC or MICo/SimSR training.

## Usage
Simply run the script:
bash
python residual_test.py

The code automatically runs two settings:

Frozen encoder and Trainable encoder

and compares:

DistanceMLP (fully parameterized) and Our Distance (structured PSD distance)

## Output

Training loss curves (Bellman residual MSE) are saved as:

fixedpoint_train_freeze.png

fixedpoint_train_notfreeze.png