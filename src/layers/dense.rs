use arrayfire::{Array, dim4, FloatingPoint, HasAfEnum, matmul, MatProp, randn};
use crate::{ForwardPass, Layer, Optimizer};

pub struct Dense<'a, T: HasAfEnum + FloatingPoint> {layer: Layer<'a, T>, weights: Array<T>, biases: Array<T>, optimizer: Optimizer}
impl<'a, T: HasAfEnum + FloatingPoint > Dense<'a, T> {
    pub fn new(neurons: u64, inputs: u64, optimizer: Optimizer) -> Self {
        Self {
            layer: Layer::default(),
            weights: randn(dim4!(inputs, neurons)),
            biases: Array::new_empty(dim4!(1, neurons)),
            optimizer
        }
    }
}

impl<'a, T: HasAfEnum + FloatingPoint> ForwardPass<'a, T> for Dense<'a, T> {
    fn forward(&mut self, x: &'a Array<T>) {
        self.layer.inputs.fill(x).ok();
        self.layer.outputs = matmul(x, &self.weights, MatProp::NONE, MatProp::NONE);
    }
}