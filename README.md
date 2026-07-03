# PAMD: Structured Adaptive Distances for Bisimulation Representations in Visual Reinforcement Learning

<p align="center">
  <b>Official repository for the ICML 2026 paper</b><br>
  <b>"PAMD: Structured Adaptive Distances for Bisimulation Representations in Visual Reinforcement Learning"</b>
</p>

<p align="center">
  <b>Daegyeong Roh*</b> · <b>Juho Bae*</b> · <b>Han-Lim Choi</b><br>
  KAIST<br>
  <sup>*</sup>Equal contribution
</p>

<p align="center">
  <a href="PAMD_paper.pdf"><b>Paper</b></a> |
  <a href="#overview"><b>Overview</b></a> |
  <a href="#method"><b>Method</b></a> |
  <a href="#repository-structure"><b>Repository Structure</b></a> |
  <a href="#results"><b>Results</b></a> |
  <a href="#citation"><b>Citation</b></a>
</p>

> **Note:** The final paper PDF is available in this repository as [`PAMD_paper.pdf`](PAMD_paper.pdf).  
> This repository includes code, training scripts, evaluation utilities, and reproduction instructions for the experiments reported in the paper.

## Overview

Many visual reinforcement learning algorithms learn representations by matching latent distances to behavioral distances induced by reward and transition similarity. However, the choice of latent distance can significantly affect downstream control performance.

Fixed, hand-designed distances such as ℓp norms can be too restrictive, while fully unconstrained pairwise distance heads may fit the metric objective without meaningfully improving the representation.

We introduce **PAMD**: **Pairwise Adaptive Mahalanobis Distance**, a structured yet adaptive latent distance for bisimulation-based representation learning. PAMD parameterizes a pair-conditioned positive-definite quadratic form, providing a more expressive alternative to fixed latent norms while avoiding the degeneracy of unconstrained distance heads.

## Method

Given two latent representations $z$ and $z'$, PAMD computes a pairwise adaptive distance

$$
d_\theta(z, z') =
\sqrt{
(z - z')^\top \widetilde{G}_\theta(z, z') (z - z') + \epsilon
}.
$$

Here, $\widetilde{G}_\theta(z, z')$ is a trace-normalized positive-definite weight matrix predicted from the latent pair.

### Positive-definite weight matrix construction via Cholesky decomposition

PAMD constructs the pair-conditioned weight matrix using a Cholesky-style decomposition. A lightweight metric network predicts the lower-triangular entries of $L_\theta(z, z')$, and the initial positive-definite matrix is formed as

$$
G_0(z, z') = L_\theta(z, z') L_\theta(z, z')^\top.
$$

This matrix is then symmetrized over the input pair, stabilized with a small ridge term, and trace-normalized. As a result, PAMD provides a structured local Mahalanobis geometry while keeping the distance tied to the latent displacement $z - z'$.

## Repository Structure

This repository contains three main components:

- `dbc`: DBC baseline and DBC with PAMD;
- `mico_simsr`: MICo and SimSR baselines with PAMD;
- `mico_residual_test`: residual-fitting diagnostics for MICo-style objectives.

Each component contains its own README with detailed setup and execution instructions.

To run DBC experiments, see [`dbc/`](dbc/).  
For MICo and SimSR experiments, see [`mico_simsr/`](mico_simsr/).  
For residual-fitting diagnostics, see [`mico_residual_test/`](mico_residual_test/).

## Results

PAMD is evaluated as a plug-in replacement for latent comparators in bisimulation-based visual reinforcement learning pipelines.

Across pixel-based DeepMind Control Suite tasks, PAMD improves downstream control performance when applied to representative operator families, including:

- DBC-style bisimulation objectives;
- independently-coupled MICo/SimSR-style objectives.

The paper also includes ablations showing that:

- full pair-conditioned quadratic forms outperform diagonal Mahalanobis variants;
- trace normalization stabilizes learning;
- PAMD is more robust under natural-video visual disturbances;
- unlike unconstrained pairwise MLP distances, PAMD preserves learning pressure on the encoder instead of fitting the fixed-point target through the distance head alone.

## Paper

The final paper is available here:

[PAMD_paper.pdf](PAMD_paper.pdf)

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{rohbae2026pamd,
  title={PAMD: Structured Adaptive Distances for Bisimulation Representations in Visual Reinforcement Learning},
  author={Roh, Daegyeong and Bae, Juho and Choi, Han-Lim},
  booktitle={Proceedings of the 43rd International Conference on Machine Learning},
  year={2026}
}
