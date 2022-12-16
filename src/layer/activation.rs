use std::sync::Arc;
use arrayfire::exp;
use crate::layer::Propagate;
use crate::Tensor;

#[derive(Clone)]
pub enum ActivationDesc {ReLU, Softmax, Sigmoid}

#[derive(Clone)]
pub struct Activation {desc: ActivationDesc, pub inputs: Arc<Tensor<f64>>, pub outputs: Arc<Tensor<f64>>, pub inputs_derivative: Arc<Tensor<f64>>}
impl Activation {
    pub fn new(desc: ActivationDesc) -> Activation {
        Activation {desc, inputs: Arc::new(Tensor::empty()), outputs: Arc::new(Tensor::empty()), inputs_derivative: Arc::new(Tensor::empty())}
    }
}

impl Propagate for Activation {
    fn forward(&mut self, x: Arc<Tensor<f64>>) {
        match &self.desc {
            ActivationDesc::ReLU => {
                self.outputs = Arc::new(Tensor::clip_min(x.as_ref(), 0.0));
            }
            ActivationDesc::Softmax => {
                let exp_values: Tensor<f64> = Tensor::exp(&(x.as_ref() - &x.max(1)));
                let probabilities: Tensor<f64> = &exp_values / &Tensor::sum(&exp_values, 1);

                self.outputs = Arc::new(probabilities);
            }
            ActivationDesc::Sigmoid => {
                self.outputs = Arc::new(1.0 / (1.0 + -&x.exp()))
            }
        }
        self.inputs = x;
    }
}
