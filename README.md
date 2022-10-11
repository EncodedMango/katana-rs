# katana-rs
A rust-based neural network library powered by arrayfire

This is a project with the goal of achieving gpu accelerated neural network computation using arrayfire to support non-cuda gpus. 
It is still hugely in development and might be so for some years to follow.

All contributions to this will be greatly appreciated.


TODO
- General
  - Layer trait

- Feedforward network
  - Activation enum
  - Loss Function
  - Optimizers
  - Dropout

- Convolutional Network
  - Convolutional Layer struct
  - Filter struct with size, stride, and padding arguments
  - Max Pooling layer
  - Other things in a convolutional network

- Recurrent Network
  - LSTM implementation (idk a lot about this type of network)


- Visualisation
  - Boring print functions for now until I have the networks figured out.
  - Cool graphs displaying how terrible a network is doing with WebGpu-rs

