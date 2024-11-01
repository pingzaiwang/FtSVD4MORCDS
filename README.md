# MATLAB Scripts for ERM-DS

Here, we provide the demo for our NeurIPS'24 paper  "*Generalized Tensor Decomposition for Understanding Multi-Output Regression under Combinatorial Shifts*" written by Andong Wang, Yuning Qiu, Mingyuan Bai, Zhong Jin, Guoxu Zhou, and Qibin Zhao. This paper is available at https://openreview.net/forum?id=1v0BPTR3AA.

This document provides instructions for using two MATLAB scripts: `test_demo_kappa.m` and `test_demo_sr_train.m`.

## test_demo_kappa.m

This script is designed to generate and test data to explore the impact of the `kappa` and `SR` parameters on the test risk. Outputs are saved in the text file `results_kappa.txt` and `results_sr_train.txt`, respectively.

### Usage
1. Ensure the MATLAB current directory includes the script's location.
2. To run the script, type `test_demo_kappa` in the MATLAB command window.

## test_demo_sr_train.m

This script focuses on training data with an emphasis on assessing how different sampling rates affect model performance.

### Usage
1. Ensure the MATLAB current directory includes the script's location.
2. To run the script, type `test_demo_sr_train` in the MATLAB command window.

## Additional Notes
- Ensure all necessary MATLAB functions and toolboxes are installed.
- Adjust script parameters as needed to fit different testing requirements.

## Bibtex Format

```
@inproceedings{
FtSVD2024NIPS,
title={Generalized Tensor Decomposition for Understanding Multi-Output Regression under Combinatorial Shifts},
author={Andong Wang, Yuning Qiu, Mingyuan Bai, Zhong Jin, Guoxu Zhou, Qibin Zhao},
booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
year={2024},
url={https://openreview.net/forum?id=1v0BPTR3AA}
}
```
