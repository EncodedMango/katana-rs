mod layer;
mod layers;
mod optimizer;

pub use layers::{Dense};
pub use layer::{Layer, ForwardPass, BackwardPass};
pub use optimizer::{Optimizer};
