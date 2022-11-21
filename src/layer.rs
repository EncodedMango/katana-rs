use lazycell::LazyCell;
use arrayfire::{Array, dim4, HasAfEnum};

pub struct Layer<'a, T: HasAfEnum> {pub inputs: LazyCell<&'a Array<T>>, pub outputs: Array<T>, pub gradient: Array<T>}
impl<'a, T: HasAfEnum> Default for Layer<'a, T> {
    fn default() -> Self {
        Self {inputs: LazyCell::default(), outputs: Array::new_empty(dim4!(1)), gradient: Array::new_empty(dim4!(1))}
    }
}

pub trait ForwardPass<'a, T: HasAfEnum> {fn forward(&mut self, x: &'a Array<T>);}
pub trait BackwardPass {fn backward(&mut self);}
