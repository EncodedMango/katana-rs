mod dense;
mod activation;

use std::sync::Arc;
pub use dense::Dense;
pub use activation::{Activation, ActivationDesc};
use crate::Tensor;

#[derive(Clone)]
pub enum LayerContainer {Dense(Dense), Activation(Activation)}
#[derive(Clone)]
pub struct Layer {layer: LayerContainer, pub inputs: Arc<Tensor<f64>>, pub outputs: Arc<Tensor<f64>>, pub inputs_derivative: Arc<Tensor<f64>>}
impl Layer {
    pub fn new(layer: LayerContainer) -> Layer {
        Layer {layer, inputs: Arc::new(Tensor::empty()), outputs: Arc::new(Tensor::empty()), inputs_derivative: Arc::new(Tensor::empty()) }
    }
}

pub trait GetLayerFields { // Trait to get certain fields from above Layer struct, when using within enum
    fn get_inputs(&self) -> Arc<Tensor<f64>>;
    fn get_outputs(&self) -> Arc<Tensor<f64>>;
}

impl GetLayerFields for Layer {
    fn get_inputs(&self) -> Arc<Tensor<f64>> {
        Arc::clone(&self.inputs)
    }

    fn get_outputs(&self) -> Arc<Tensor<f64>> {
        Arc::clone(&self.outputs)
    }
}

impl Propagate for Layer {
    fn forward(&mut self, x: Arc<Tensor<f64>>) {
        match &mut self.layer {
            LayerContainer::Dense(dense) => {
                dense.forward(Arc::clone(&x));
                self.outputs = Arc::clone(&dense.outputs);
            }
            LayerContainer::Activation(activation) => {
                activation.forward(Arc::clone(&x));
                self.outputs = Arc::clone(&activation.outputs)}
        }
        self.inputs = Arc::clone(&x);
    }
}

pub trait Propagate {
    fn forward(&mut self, x: Arc<Tensor<f64>>);
}
