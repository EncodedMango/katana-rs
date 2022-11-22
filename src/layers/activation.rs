use arrayfire::{Array, ConstGenerator, HasAfEnum, lt, selectl};
use crate::{Layer, Propagate};

pub enum ActivationDescriptor {ReLU, Softmax, Sigmoid, Tanh, LeakyReLU(f32)}
pub struct Activation<'a, T: HasAfEnum> {descriptor: ActivationDescriptor, pub layer: Layer<'a, T>}
impl<'a, T: HasAfEnum> Activation<'a, T> {
    pub fn new(descriptor: ActivationDescriptor) -> Self {
        Self {descriptor, layer: Layer::default()}
    }
}

impl<'a, T: HasAfEnum + Default + Clone + ConstGenerator<OutType = T>> Propagate<'a, T> for Activation<'a, T> {
    fn forward(&mut self, x: &'a Array<T>) {
        match self.descriptor {
            ActivationDescriptor::ReLU => { // Rectified Linear activation function
                self.layer.inputs.fill(x).ok();
                self.layer.outputs = selectl(0.0, &lt(x, &T::default(), true), x);
            }
            ActivationDescriptor::Softmax => {}
            ActivationDescriptor::Sigmoid => {}
            ActivationDescriptor::Tanh => {}
            ActivationDescriptor::LeakyReLU(_) => {}
        }
    }

    fn backward(&mut self) {
        todo!()
    }
}