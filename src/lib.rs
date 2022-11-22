mod layer;
mod layers;
mod optimizer;

pub use layers::{Dense, ActivationDescriptor, Activation};
pub use layer::{Layer, Propagate};
pub use optimizer::{Optimizer};
