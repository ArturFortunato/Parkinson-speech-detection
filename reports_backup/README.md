## What is this?

Each folder contains the results of one set of experiments

### Baseline

Each dataset is used independently

### Semi

Each model's training set is composed of one dataset and a subset of another. 
The model's test set is the remaining part of the second dataset.

### Independent

Each model is trained with two datasets and tested with the third

## File nomenclature

<trainPercentage>\_<alpha>\_<maxIterations>\_<actFunction>.txt where

1. __trainPercentage__ is the percentage of the dataset used for testing
2. __alpha__ is the L2 penalty (regularization term) parameter of the MLP
3. __maxIterations__ is the maximum number of iterations accepted until training stops
4. __actFunction__ is the activation function for the hidden and output layers of the MLP

## Model's architecture

For this section __n__ refers to the number of features

### Baseline/Semi/Independent

MLP with an input layer with __n__ nodes, a hidden layer with __n + 1__ nodes and an output layer with __1__ node


### Baseline/Semi/Independent[\_200]

MLP with an input layer with __n__ nodes, 2 hidden layers with __200__ nodes, and an output layer with __1__ node

