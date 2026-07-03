# DBC / DBC with our distance

This repository contains minimal training code for DBC, and DBC with Our Distance. 


## Requirements
- CUDA 12.x–compatible GPU
- Conda

## To create the conda environment:
```bash
conda env create -f conda_env.yml

## To Activate the environment:

conda activate dbc

## Training
Run training via bash:

bash run.sh

## Switching agents and settings
Edit run.sh to change:

--agent: select an agent from the agents/ directory

task configuration: --domain_name, --task_name, --seed, etc.

## Outputs

Logs, videos, and TensorBoard files are saved under --work_dir (default: ./log).