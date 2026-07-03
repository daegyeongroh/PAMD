# Code release

This repository contains three components:

* `dbc`: DBC baseline and DBC with our distance
* `mico_simsr`: MICo and SimSR baselines with our distance
* `mico_residual_test`: residual-fitting diagnostics for MICo-style objectives

Each component has its own README with detailed instructions.

## Third-party code

This repository builds on the following public research codebases:

* `dbc/` is based on Facebook Research's DBC implementation:
  https://github.com/facebookresearch/deep_bisim4control
  The original project is licensed under CC BY-NC 4.0.

* `mico_simsr/` is based on the public SimSR implementation:
  https://github.com/bit1029public/SimSR

We thank the original authors for releasing their code. Our modifications add the proposed structured adaptive distance and the experiments used in our submission.
