use std::path::Path;
use arrayfire::{af_print, Array, dim4, sum};
use katana::{Layer, Dataset, DenseLayer, ReLUActivation, SoftmaxActivation};

fn main() {
    let dataset: Dataset<f32> = Dataset::from_csv(&Path::new(&"examples/spiral.csv")).unwrap();

    let dense1: DenseLayer<f32> = DenseLayer::new(2, 3);
    let dense2: DenseLayer<f32> = DenseLayer::new(3, 3);

    let activation1: ReLUActivation<f32> = ReLUActivation { dinputs: Array::new_empty(dim4!(1)) };
    let activation2: SoftmaxActivation<f32> = SoftmaxActivation { dinputs: Array::new_empty(dim4!(1)) };

    let mut output = dense1.forward(&dataset.x);
    output = activation1.forward(&output);
    output = dense2.forward(&output);
    output = activation2.forward(&output);

    af_print!("{}", sum(&output, 1)); // seems about right
}