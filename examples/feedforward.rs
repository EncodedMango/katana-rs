use arrayfire::{Array, dim4, randn};
use katana::{Dense, ForwardPass, Layer, Optimizer};

fn main() {
    let mut dense1: Dense<f32> = Dense::new(3, 5, Optimizer::None);

    let inputs: Array<f32> = randn(dim4!(5, 5));
    dense1.forward(&inputs);
}