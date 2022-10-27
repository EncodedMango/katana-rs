use arrayfire::{af_print, Array, dim4, FloatingPoint, HasAfEnum, matmul, MatProp, randn, tile};
use crate::Layer;

pub struct DenseLayer<T: HasAfEnum> {weights: Array<T>, biases: Array<T>}

impl<T: HasAfEnum + FloatingPoint> DenseLayer<T> {
    pub fn new(n_inputs: u64, n_neurons: u64) -> Self {
        Self {weights: randn(dim4!(n_inputs, n_neurons)), biases: Array::new_empty(dim4!(1, n_neurons))}
    }
}

impl<T: HasAfEnum + FloatingPoint> Layer<T> for DenseLayer<T> {
    fn forward(&self, inputs: &Array<T>) -> Array<T> {
        matmul(&inputs, &self.weights, MatProp::NONE, MatProp::NONE) + &tile(&self.biases, dim4!(inputs.dims().get()[0]))
    }
    fn backward(&self, inputs: &Array<T>) -> Array<T> {
        todo!()
    }
}