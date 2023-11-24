# PTP-DA
Official PyTorch code for the paper "Adaptive Pedestrian Trajectory Prediction via Target-Directed Data Augmentation". 

>Abstract: Pedestrian trajectory prediction is an important task for many applications such as autonomous driving and surveillance systems. Yet the prediction performance drops dramatically when applying a model trained on the source domain to a new target domain. Therefore, it is of great importance to adapt a predictor to a new domain. Previous works mainly focus on feature-level alignment to solve this problem. In contrast, we solve it from a new perspective of instance-level alignment. Specifically, we first point out one key factor of the domain gaps, i.e., trajectory angles, and then augment the source training data by target-directed orientation augmentation so that its distribution matches with that of the target data. In this way, the trajectory predictor trained on the aligned source data performs better on the target domain. Experiments on standard baselines show that our method improves the state of the art by a large margin.

# Installation

## Environment

OS: Linux / RTX 2080Ti
Python == 3.7.9
PyTorch == 1.7.1+cu110

## Dependencies

Install the dependencies from the requirements.txt:

```bash
pip install -r requirements.txt
```
