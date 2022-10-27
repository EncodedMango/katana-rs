mod dense;
mod activation;

use arrayfire::{Array, HasAfEnum};
pub use dense::DenseLayer;
pub use activation::{ReLUActivation, SoftmaxActivation};

pub trait Layer<T: HasAfEnum> {
    fn forward(&self, inputs: &Array<T>) -> Array<T>; // Forward pass
    fn backward(&self, inputs: &Array<T>) -> Array<T>; // Backward pass
}