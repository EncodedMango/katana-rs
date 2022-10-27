use std::f64::consts::E;
use arrayfire::{af_print, Array, dim4, exp, HasAfEnum, max, sum, tile, transpose};
use crate::Layer;

pub struct SoftmaxActivation<T: HasAfEnum> {pub dinputs: Array<T>}
impl<T: HasAfEnum> Layer<T> for SoftmaxActivation<T> {
    fn forward(&self, inputs: &Array<T>) -> Array<T> {
        let exp_values: Array<T> = exp(&(inputs - &tile(&max(inputs, 1).cast(), dim4!(1, inputs.dims().get()[1])))).cast();
        &exp_values / tile(&sum(&exp_values, 1), dim4!(1, inputs.dims().get()[1])).cast()
    }
    fn backward(&self, inputs: &Array<T>) -> Array<T> {
        todo!()
    }
}