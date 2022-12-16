use std::sync::Arc;
use crate::dim4;
use crate::Tensor;
use crate::layer::Propagate;

#[derive(Clone)]
pub struct Dense {pub inputs: Arc<Tensor<f64>>, pub outputs: Arc<Tensor<f64>>, pub inputs_derivative: Arc<Tensor<f64>>, weights: Tensor<f64>, biases: Tensor<f64>, weights_derivative: Tensor<f64>, biases_derivative: Tensor<f64>}
impl Dense {
    pub fn new(inputs: u64, neurons: u64) -> Self {
        Self {inputs: Arc::new(Tensor::empty()), outputs: Arc::new(Tensor::empty()), inputs_derivative: Arc::new(Tensor::empty()), weights: Tensor::randn(dim4!(inputs, neurons)), biases: Tensor::empty_d(dim4!(1, neurons)), weights_derivative: Tensor::empty(), biases_derivative: Tensor::empty()}
    }
}

impl Propagate for Dense {
    fn forward(&mut self, x: Arc<Tensor<f64>>) {
        self.outputs = Arc::new(&Tensor::dot(x.as_ref(), &self.weights) + &self.biases);
        self.inputs = x;
    }
}
