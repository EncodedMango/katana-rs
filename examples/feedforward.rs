use std::path::Path;
use arrayfire::{af_print, Array, dim4, sum};
use katana::{Activation, Layer, Dataset, DenseLayer, BiographicalCrossEntropy};

fn main() {
    let dataset: Dataset<f32> = Dataset::from_csv(&Path::new(&"examples/spiral.csv")).unwrap();

    let dense1: DenseLayer<f32> = DenseLayer::new(2, 3);
    let dense2: DenseLayer<f32> = DenseLayer::new(3, 3);

    let activation1: Activation = Activation::ReLU;
    let activation2: Activation = Activation::Softmax;

    let loss: BiographicalCrossEntropy<f32> = BiographicalCrossEntropy {placeholder: Array::new_empty(dim4!(0))};

    let mut output = dense1.forward(&dataset.x);
    output = activation1.forward(&output);
    output = dense2.forward(&output);
    output = activation2.forward(&output);
    //print!("{}", loss.calculate(&output, &dataset.y));

    af_print!("{}", output); // seems about right
}