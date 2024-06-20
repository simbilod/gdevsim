# gdevsim 0.0.1

> :warning: **Warning**: This project is in early development stages and may be subject to significant changes.

A simplified interface (enabled by gdsfactory) and collection of models to the open-source TCAD simulator DEVSIM.

Features:
* Easily define simulations from gdsfactory primitives (Components, LayerStacks, Ports)
* Seamlessly simulate devices from existing gdsfactory PDKs
* Repository of documented semiconductor models
* Adaptive mesh refinement
* Automatic file management; manipulate all inputs and outputs from Python
* Hassle-free installation: `pip install gdevsim`
* Compatible with [Ray](https://github.com/ray-project/ray) for seamless distributed computing (local/cluster/cloud), lazy parallelism, and optimization

The aim is to sacrifice some flexibility for simpler automation. To setup more fine-grained simulations, DEVSIM can be used directly.

## Usage

See the documentation for more information on the API or for examples.

The CAD backend is gdsfactory. The meshing backend is meshwell, which wraps gmsh. The solver backend is DEVSIM.

## Installation

For users:

`pip install gdevsim`

For contributors:

`pip install gdevsim[dev]`

## Contributing and licensing

Contributions welcome! To contribute, fork the repository, make your changes in a new branch, and open a pull request. Make sure `pre-commit` is run to format your code, and make sure all tests pass.

DEVSIM is licensed under the Apache 2.0 license, and so is this repository. Contributions will be licensed accordingly.

## Acknowledgements

* Simon Bilodeau (Princeton): maintainer, initial implementation, surface recombination model, bandgap narrowing model
* Juan Sanchez (DEVSIM): for developing and maintaining DEVSIM, useful scripts, useful examples, many models, and assistance
* Gerrit Lükens (pmdtechnologies): for [optical generation model](https://forum.devsim.org/t/ssac-and-transient-in-generation-term/229)
* Joaquin Matres Abril (Google) & gdsfactory community: for maintaining and developing gdsfactory/gplugins