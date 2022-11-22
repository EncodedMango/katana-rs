use std::time::Instant;
use arrayfire::{Array, dim4, info, print, randn, set_device};
use katana::{Dense, Propagate, Layer, Optimizer, Activation, ActivationDescriptor};

fn main() {
    info();
    set_device(1);

    let mut dense1: Dense<f32> = Dense::new(3, 2, Optimizer::None);
    let mut activation1: Activation<f32> = Activation::new(ActivationDescriptor::ReLU);
    let mut dense2: Dense<f32> = Dense::new(3, 3, Optimizer::None);

    let inputs: Array<f32> = randn(dim4!(10000, 2));

    let t: Instant = Instant::now();

    //println!("[{:?}] Entry point", t.elapsed());
    //print(&inputs);

    dense1.forward(&inputs);

    //println!("[{:?}] Forward pass", t.elapsed());
    //print(&dense1.layer.outputs);

    activation1.forward(&dense1.layer.outputs);
    //println!("[{:?}] First activation pass", t.elapsed());
    //print(&activation1.layer.outputs);

    dense2.forward(&activation1.layer.outputs);
    println!("[{:?}] Second forward pass", t.elapsed());
    //print(&dense2.layer.outputs);
}