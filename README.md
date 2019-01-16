# Tiny Machine Learning System

## Reimplementation
this project is ready to reimplement since the low efficiency, incomplete api and inappropriate uses of eigen api.

## About
tiny-machine-learning-system is about: 
C++ implementations of some of the fundamental Machine Learning models and algorithms from scratch
and high-level python implementations of calling the C functions.

The purpose of this project is not to produce as optimized and computationally efficient algorithms as possible but rather to present the inner workings of them in a transparent and accessibel way.

## Table of Contents
- [Tiny Machine Learning System](#tiny-machine-learning-system)
  *  [About](#about)
  *  [Table of Contents](#table-of-contents)
  *  [Project structure](#project-structure)
  *  [Supported algorithms](#supported-algorithms)
  *  [Dependencies](#dependencies)
  *  [Examples](#examples)

## Project structure
    .
    ├── src                         # C++ implementations
    ├── examples                    # examples of C++ implementations
    ├── wrapper                     # c interface by wrapping C++ implementations
    ├── python-package              # high-level python implementations
    ├── generate..dataset.py        # generate testing samples
    └── makefile                    # makefile for c++ implementations

## Supported algorithms
only support C++ implementations currently since the high-level python implementations still have some bugs.
### C++ implementations
* Logistic Regression
* k nearest neighbors
* naive bayes
* decision tree
* support vector machine
* k means
* perceptron
* linear discriminant analysis
* principal component analysis

### Python implementations
* Logistic Regression

## Dependencies
* Eigen

## Examples
	git clone https://github.com/KaiminLai/tiny-machine-learning-system
	cd tiny-machine-learning-system
	python generate_classification_dataset.py

modify the makefile to test c++ implementations respectively.
