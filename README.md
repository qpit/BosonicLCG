# BosonicLCG

A Python-based simulator for continuous variable circuits with Gaussian components and photon-number resolving detectors. This is an extension of the [bosonic backend](https://strawberryfields.ai/photonics/demos/run_intro_bosonic.html) from strawberryfields with an improved representation of non-Gaussian states in the linear combination of Gaussians framework, which allows the user to e.g. simulate state preperation with GBS circuits.

## Dependencies
- See `pyproject.toml`

## Installation
I recommend using the [uv package manager](https://docs.astral.sh/uv/).

`uv add "lcg_plus @ git+https://github.com/qpit/lcg_plus.git"`

## Features
- Simulation of phase-space Wigner functions in the Gaussian representation by tracking covariance matrices, displacement vectors, and weights.
- Non-Gaussian states and measurement POVM elements are represented as linear combinations of multivariate Gaussians.

Please see our preprint for more information.
>Olga Solodovnikova, Ulrik L. Andersen, Jonas S. Neergaard-Nielsen "Fast simulations of continuous-variable circuits using the coherent state decomposition" [arXiv:2508.06175 \[quant-ph\]](http://arxiv.org/abs/2508.06175)

## Tutorials
See Jupyter notebooks in `demos/`. To run the notebooks,

`git clone git@github.com:qpit/lcg_plus.git`

`uv pip install .` from root.

Remember to add the kernel

`uv run python -m ipykernel install --user --name=lcg_plus --display-name "Python (lcg_plus)"`

`uv run jupyter-lab`
