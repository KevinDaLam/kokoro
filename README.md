
<p align="center"><a href="https://github.com/KevinDaLam/kokoro" target="_blank"><img width="75" src="https://raw.githubusercontent.com/KevinDaLam/kokoro/master/img/logo.jpg"></a></p>

<h1 align="center">Kokoro</h1>

<p align="center">An artificial neural network module driven by <a href="https://github.com/numpy/numpy/" target="_blank">NumPy</a></p>

<p align="center">
  <a href="https://travis-ci.org/KevinDaLam/kokoro"><img src="https://travis-ci.org/KevinDaLam/kokoro.svg?branch=master" alt="Build Status"></a>
</p>

### Usage

Kokoro is a neural network class that is intended to be flexible for any numerical input/output vector. 

```python
# Train and Predict
import kokoro

ANN = kokoro.ANNetwork(learning_rate, n_input_neurons, n_hidden_neurons, n_hidden_layers, n_output_neurons)

#Input and output data are np.matrix type
ANN.Train(input_data, output_data, n_iterations)

ANN.Predict(input_data)

```

### Motivation

This module was made for the purpose of numerical analysis of basic neural network principles, especially in regards to the backpropagation algorithm and its relationship with gradient descent. Understanding neural networks is a great introduction to machine learning concepts under supervised learning and this project allowed me to gain more experience with NumPy and its linear algebra capabilities.

### References

[Machine Learning - Andrew Ng](https://www.coursera.org/learn/machine-learning/)
[A Step by Step Backpropagation Example - Matt Mazur](https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/)
