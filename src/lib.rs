// General (applies to all networks)
// TODO: Layer trait
// TODO: Dataset struct (kinda done (with it))

// Feedforward network
// TODO: Activation enum
// TODO: Loss Function
// TODO: Optimizers
// TODO: Dropout

// Convolutional Network
// TODO: Convolutional Layer struct
// TODO: Filter struct with size, stride, and padding arguments
// TODO: Max Pooling layer
// TODO: Other things in a convolutional network

// Recurrent Network
// TODO: LSTM implementation (idk a lot about this type of network)


// Visualisation
// TODO: Boring print functions for now until I have the networks figured out.
// TODO: Cool graphs displaying how terrible a network is doing with WebGpu-rs

mod layer;
mod dataset;

pub use dataset::Dataset;
pub use layer::{Layer, DenseLayer, ReLUActivation, SoftmaxActivation};