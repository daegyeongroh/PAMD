# MICo / SimSR / MICO-SimSR with Our Distance

This repository contains minimal training code for MICo, SimSR, and MICO-SimSR with Our Distance. 

## Requirements
- CUDA 12.x–compatible GPU
- Conda

## To create the conda environment:
```bash
conda env create -f conda_env.yml

## To Activate the environment:

conda activate micosimsr

## Training
Run training via bash:

bash run.sh

## Switching agents and settings
Edit run.sh to change:

--agent: select an agent from the agents/ directory

task configuration: --domain_name, --task_name, --seed, etc.

## Outputs

Logs and checkpoints are saved under --work_dir (default: ./tmp).