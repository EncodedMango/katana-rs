use arrayfire::{Array, ConstGenerator, gt, HasAfEnum};
use crate::Layer;

pub struct ReLUActivation<T: HasAfEnum> {pub dinputs: Array<T>}
impl<T: HasAfEnum + Default + ConstGenerator<OutType = T> + Clone> Layer<T> for ReLUActivation<T> {
    fn forward(&self, inputs: &Array<T>) -> Array<T> {
        gt(inputs, &T::default(), true).cast::<T>() * inputs
    }
    fn backward(&self, inputs: &Array<T>) -> Array<T> {
        todo!()
    }
}