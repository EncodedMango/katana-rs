use std::sync::Arc;
use arrayfire::{HasAfEnum, print};
use crate::Tensor;

#[derive(Clone)]
pub enum LossDesc {CrossEntropy}

#[derive(Clone)]
pub struct Loss {desc: LossDesc, inputs: Arc<Tensor<f64>>, outputs: Arc<Tensor<f64>>, inputs_derivative: Arc<Tensor<f64>>, y: Arc<Tensor<bool>>}
impl Loss {
    pub fn new(desc: LossDesc) -> Loss {
        Loss {desc, inputs: Arc::new(Tensor::empty()), outputs: Arc::new(Tensor::empty()), inputs_derivative: Arc::new(Tensor::empty()), y: Arc::new(Tensor::empty())}
    }
}

pub trait PropagateLoss {
    fn forward(&mut self, x: Arc<Tensor<f64>>, y: Arc<Tensor<bool>>) -> f64;
    fn backward(&mut self, x: Arc<Tensor<f64>>);
}

impl PropagateLoss for Loss {
    fn forward(&mut self, x: Arc<Tensor<f64>>, y: Arc<Tensor<bool>>) -> f64{
        let x_clipped: Tensor<f64> = x.clip(0.0001, 0.9999);
        let correct_confidences: Tensor<f64> = (&x_clipped * y.as_ref()).sum(1);

        let mut v: Vec<f64> = vec![0.0];


        match &self.desc {
            LossDesc::CrossEntropy => {
                self.outputs = Arc::new(-(correct_confidences.log()));
                self.outputs.mean(0).host(&mut v)}
        }

        v[0]
    }

    fn backward(&mut self, x: Arc<Tensor<f64>>) {
        match &self.desc {
            LossDesc::CrossEntropy => {
                self.inputs_derivative = Arc::new(&((!self.y.as_ref()).cast::<f64>()) / x.as_ref() / x.dims()[0] as f64)
            }
        }
    }
}