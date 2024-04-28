# Sequence-Space Jacobians in Continuous Time

This repository hosts the code for the research paper titled "Sequence-Space Jacobians in Continuous Time" by R. Glawion (2023), available at SSRN 4504829. The paper builds upon the work of Auclert et al. (2021) presented in "Using the sequence‐space Jacobian to solve and estimate heterogeneous‐agent models" published in Econometrica.

## Citation

If you use this code for your research, please consider citing the original papers:

- Glawion, R. (2023). Sequence-Space Jacobians in Continuous Time. Available at SSRN 4504829.

- Auclert, A., Bardóczy, B., Rognlie, M., & Straub, L. (2021). Using the sequence‐space Jacobian to solve and estimate heterogeneous‐agent models. Econometrica, 89(5), 2375-2408.

## Overview

The code in this repository implements the methods described in Glawion (2023), which extends the sequence‐space Jacobian framework to continuous time. It is based on the foundational work presented by Auclert et al. (2021) and provides implementations for various models and algorithms.

## Models and Scripts

- `CanonicalHANK.py`: Implements the canonical one-asset HANK model in a continuous-time framework, building upon the work of Auclert et al. (2021)

- `KrusellSmithSSJ.py`: Implements the sequence-space Jacobian algorithm for the Krussel-Smith model in continuous time

- `FakeNewsExampleHugget.py`: Illustrates the Fake-News Algorithm for a continuous-time Hugget model, following the approach outlined by Auclert et al. (2021)

- `estimation.py`: Illustrates the adaptation of methods and functions from the discrete-time toolbox to the continuous-time framework, enabling estimation of continuous-time models.

- `ComparisonDTandCTHank.py`: Compares discrete and continuous-time implementations of the one-asset HANK model.

## Usage

To use the code in this repository, follow these steps:

1. Clone the repository to your local machine (only needed for the estimation part):
git clone https://github.com/shade-econ/sequence-jacobian.git

2. Install the necessary dependencies.

3. Explore the provided scripts and models.

4. Refer to the corresponding papers for detailed explanations of the models and algorithms implemented.