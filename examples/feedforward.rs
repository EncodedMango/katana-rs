use std::sync::Arc;
use arrayfire::{dim4, info, print, randn, set_device, Array, Window, MarkerType, ColorMap, Seq, range, col, sin, cos, join, assign_seq, constant, tile, iota, transpose, moddims, Dim4};
use katana::{Tensor, layer, Model, PropagateModel, OptimizerDesc, Loss, LossDesc, PropagateLoss};
use katana::layer::{Activation, ActivationDesc, Dense, Layer, LayerContainer, Propagate};

fn create_spiral_dataset(samples: u64, classes: u64) -> (Arc<Tensor<f64>>, Arc<Tensor<bool>>) {
    let mut x: Array<f64> = Array::new_empty(dim4!(samples * classes, 2));
    let mut y: Array<bool> = Array::new_empty(dim4!(samples * classes, classes));

    for class_number in 0..classes {
        let ix: Seq<u32> = Seq::new((samples * class_number) as u32, (samples * (class_number + 1)) as u32 - 1, 1);

        let r: Array<f64> = range::<f64>(dim4!(samples), 0) / (samples - 1) as f64;
        let t: Array<f64> = ((class_number as f64 * 4.2) + (&r * 4.2)) + (randn::<f64>(dim4!(samples)) * 0.1);

        let xy: Array<f64> = join(1, &(&r * sin(&(&t * 2.5))), &(r * cos(&(t * 2.5))));
        assign_seq(&mut x, &[ix, Seq::new(0, 1, 1)], &xy);
        assign_seq(&mut y, &[ix, Seq::new(class_number as u32, class_number as u32, 1)], &constant(true, dim4!(samples)));
    }

    (Arc::new(Tensor::from(x)), Arc::new(Tensor::from(y)))
}
fn create_image(d: u32, model: &Model) -> Tensor<f64> {
    let mut image = Tensor::empty_d(dim4!(d as u64, d as u64, 3));

    let y_s: Tensor<f64> = Tensor::from(range::<f64>(dim4!(d as u64), 0) / d - (range::<f64>(dim4!(d as u64), 0) / d * 2));

    let mut model = model.clone();

    for i in 0..d {
        let x_s = Tensor::from(constant((i as f64 / d as f64 * 2.0) - 1.0, dim4!(d as u64)));
        let xy = Arc::new(Tensor::from(join(1, x_s.as_ref(), y_s.as_ref())));

        model.forward(Arc::clone(&xy));

        image.assign_seq(&[Seq::new(i, i, 1), Seq::new(0, d - 1, 1), Seq::new(0, 2, 1)], &model.l_output.moddims(dim4!(1, d as u64, 3)));
    }
    image
}

fn view_tensor_image(window: &Window, img: &Tensor<f64>) {
    window.draw_image(&img.as_ref().cast::<f32>(), Option::None);
}

fn main() {
    info();

    let (x, y) = create_spiral_dataset(100, 3);
    let mut image: Tensor<f64> = Tensor::empty_d(dim4!(256, 256, 3));

    let mut dense1: Layer = Layer::new(LayerContainer::Dense(Dense::new(2, 3)));
    let mut activation1: Layer = Layer::new(LayerContainer::Activation(Activation::new(ActivationDesc::ReLU)));
    let mut dense2: Layer = Layer::new(LayerContainer::Dense(Dense::new(3, 3)));
    let mut activation2: Layer = Layer::new(LayerContainer::Activation(Activation::new(ActivationDesc::Softmax)));

    let mut model: Model = Model::new();
    model.add_vec(vec![dense1, activation1, dense2, activation2]);

    model.forward(x);
    model.calculate_loss(y);

    println!("Loss: {}", model.loss);

    let image = create_image(256, &model);
    let window = Window::new(512, 512, "Feedforward".to_string());

    loop {
        view_tensor_image(&window, &image);
        if window.is_closed() {break}
    }
}
