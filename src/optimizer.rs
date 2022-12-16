#[derive(Clone)]
pub enum OptimizerDesc {None}
#[derive(Clone)]
pub struct Optimizer {
    desc: OptimizerDesc
}
impl Optimizer {
    pub fn new(desc: OptimizerDesc) -> Optimizer {Optimizer {desc}}
}