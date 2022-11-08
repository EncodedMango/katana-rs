use arrayfire::{Array, ConstGenerator, dim4, exp, gt, HasAfEnum, max, sum, tile};
use crate::Layer;

pub enum Activation {ReLU, Softmax}
impl<T: HasAfEnum + Default + ConstGenerator<OutType = T> + Clone> Layer<T> for Activation {
    fn forward(&self, inputs: &Array<T>) -> Array<T> {
        return match self {
            Activation::ReLU => { // Rectified linear activation function
                gt(inputs, &T::default(), true).cast::<T>() * inputs}
            Activation::Softmax => { // Softmax activation function
                let exp_values: Array<T> = exp(&(inputs - &tile(&max(inputs, 1).cast(), dim4!(1, inputs.dims().get()[1])))).cast();
                &exp_values / tile(&sum(&exp_values, 1), dim4!(1, inputs.dims().get()[1])).cast()}
        }

    }

    fn backward(&self, inputs: &Array<T>) -> Array<T> {
        todo!()
    }
}