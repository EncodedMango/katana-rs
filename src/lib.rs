use std::fmt::{Debug, Formatter};
use std::ops::{Add, Deref, DerefMut, Div, Mul, Neg, Not, Sub};
use std::process::Output;
use arrayfire::{Array, Dim4, max, exp, HasAfEnum, InterpType, moddims, resize, tile, sum, print, matmul, MatProp, randn, selectl, lt, Seq, assign_seq, range, transpose, index, FloatingPoint, constant, ConstGenerator, ImplicitPromote, gt, log, mean};
pub mod layer;
use layer::Layer;
pub use arrayfire::dim4;
mod model;
mod optimizer;
mod loss;

pub use loss::{Loss, LossDesc, PropagateLoss};
pub use optimizer::{Optimizer, OptimizerDesc};
pub use model::{Model, PropagateModel};

#[derive(Clone)]
pub struct Tensor<T: HasAfEnum> {a: Array<T>} // Tensor struct that is a handle to an Array

pub enum TensorErr {BroadCastShapeMismatch}
impl Debug for TensorErr {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match &self {
            TensorErr::BroadCastShapeMismatch => write!(f, "Can't broadcast Tensor because shapes can't be broadcast")
        }
    }
}

impl<T: HasAfEnum + FloatingPoint> Tensor<T> {
    pub fn dot(&self, rhs: &Tensor<T>) -> Tensor<T> { // Dot function for vec-vec matrix-vec matrix-matrix products
        return if self.numdims() == rhs.numdims() {
            Tensor::from(matmul(&self.a, &rhs.a, MatProp::NONE, MatProp::NONE))
        } else {
            Tensor::dot(&self, &rhs.broadcast(self.dims()).unwrap())
        }
    }
}

impl Tensor<f64> {
    pub fn clip(&self, min: f64, max: f64) -> Tensor<f64> {
        let min_clipped: Array<f64> = selectl(min, &lt(&self.a, &min, false), &self.a);
        Tensor::from(selectl(max, &gt(&min_clipped, &max, true), &min_clipped))
    }
    pub fn clip_min(&self, min: f64) -> Tensor<f64> {
        Tensor::from(selectl(min, &lt(&self.a, &min, true), &self.a))
    }
    pub fn randn(dim: Dim4) -> Tensor<f64> {
        Tensor::from(randn::<f64>(dim))
    }
}

impl<T: HasAfEnum> Tensor<T> {
    pub fn numdims(&self) -> u32 {self.a.numdims()}
    pub fn dims(&self) -> Dim4 {self.a.dims()}
    pub fn exp(&self) -> Tensor<T> {
        Tensor::from(exp(&self.a).cast())
    }
    pub fn log(&self) -> Tensor<T> {
        Tensor::from(log(&self.a).cast())
    }
    pub fn mean(&self, dim: i64) -> Tensor<T> {
        Tensor::from(mean(&self.a, dim).cast())
    }
    pub fn host(&self, v: &mut [T]) {
        self.a.host(v);
    }
    pub fn max(&self, dim: i32) -> Tensor<T> {
        Tensor::from(max(&self.a, dim).cast())
    }
    pub fn sum(&self, dim: i32) -> Tensor<T> {
        Tensor::from(sum(&self.a, dim).cast())
    }
    pub fn empty() -> Tensor<T> {
        Tensor {a: Array::new_empty(dim4!(1))}
    }
    pub fn empty_d(dim: Dim4) -> Tensor<T> {
        Tensor::from(Array::new_empty(dim))
    }

    pub fn as_ref(&self) -> &Array<T> { &self.a }
    pub fn transpose(&self, conjugate: bool) -> Tensor<T> {
        Tensor::from(transpose(&self.a, conjugate))
    }
    pub fn index(&self, x: &[Seq<u32>]) -> Tensor<T> {
        Tensor::from(index(&self.a, x))
    }
    pub fn moddims(&self, n_dims: Dim4) -> Tensor<T> {
        Tensor::from(moddims(&self.a, n_dims))
    }

    pub fn assign_seq(&mut self, seqs: &[Seq<u32>], rhs: &Tensor<T>) {
        assign_seq(&mut self.a, seqs, &rhs.a);
    }
    pub fn from(a: Array<T>) -> Tensor<T> {
        Tensor {a}
    }
    pub fn cast<U: HasAfEnum>(&self) -> Tensor<U> {
        Tensor::from(self.a.cast::<U>())
    }

    pub fn broadcast(&self, dims: Dim4) -> Result<Tensor<T>, TensorErr> {
        let mut can_be_broadcast: bool = true;

        let mut tile_dims: Dim4 = Dim4::new(&[0, 0, 0, 0]);
        for i in 0..4 {
            if self.a.dims()[3 - i] == dims[3 - i] {tile_dims[3 - i] = 1}
            else if self.a.dims()[3 - i] == 1 {tile_dims[3 - i] = dims[3 - i]}
            else if dims[3 - i] == 1 {tile_dims[3 - i] = self.dims()[3 - i]}
            else {can_be_broadcast = false}
        }

        if can_be_broadcast {
            return Ok(Tensor::from(tile(&self.a, tile_dims)))
        }

        Err(TensorErr::BroadCastShapeMismatch)
    }
}

// Numpy-esque Operators
impl<T: HasAfEnum> Add for &Tensor<T> {
    type Output = Tensor<T>;
    fn add(self, rhs: Self) -> Self::Output {
        return if self.dims() == rhs.dims() {Tensor::from(&self.a + &rhs.a)}
        else {self + &rhs.broadcast(self.dims()).unwrap()}
    }
}
impl<T: HasAfEnum> Sub for &Tensor<T> {
    type Output = Tensor<T>;
    fn sub(self, rhs: Self) -> Self::Output {
        return if self.dims() == rhs.dims() {Tensor::from(&self.a - &rhs.a)}
        else {
            self - &rhs.broadcast(self.dims()).unwrap()
        }
    }
}
impl<T: HasAfEnum> Div for &Tensor<T> {
    type Output = Tensor<T>;
    fn div(self, rhs: Self) -> Self::Output {
        return if self.dims() == rhs.dims() {Tensor::from(&self.a / &rhs.a)}
        else {
            self / &rhs.broadcast(self.dims()).unwrap()
        }
    }
}
impl<T: HasAfEnum> Mul for &Tensor<T> {
    type Output = Tensor<T>;
    fn mul(self, rhs: Self) -> Self::Output {
        return if self.dims() == rhs.dims() {Tensor::from(&self.a * &rhs.a)}
        else {self * &rhs.broadcast(self.dims()).unwrap()}
    }
}

impl Mul<&Tensor<bool>> for &Tensor<f64> {
    type Output = Tensor<f64>;
    fn mul(self, rhs: &Tensor<bool>) -> Self::Output {
        return if self.dims() == rhs.dims() {Tensor::from(&self.a * &rhs.a)}
        else {self * &rhs.broadcast(self.dims()).unwrap()}
    }
}

impl<T: HasAfEnum + Default + ConstGenerator<OutType = T>> Neg for &Tensor<T> {
    type Output = Tensor<T>;
    fn neg(self) -> Self::Output {
        Tensor::from(constant(T::default(), self.dims()) - self.a.copy())
    }
}
impl Not for Tensor<bool> {
    type Output = Tensor<bool>;
    fn not(self) -> Self::Output {
        Tensor::from(&self.a * false)
    }
}
impl Not for &Tensor<bool> {
    type Output = Tensor<bool>;
    fn not(self) -> Self::Output {
        Tensor::from(&self.a * false)
    }
}


impl<T: HasAfEnum + Default + ConstGenerator<OutType = T>> Neg for Tensor<T> {
    type Output = Tensor<T>;
    fn neg(self) -> Self::Output {
        Tensor::from(constant(T::default(), self.dims()) - self.a.copy())
    }
}

impl<T: HasAfEnum + Clone + ConstGenerator<OutType = T>> Add<T> for Tensor<T> {
    type Output = Tensor<T>;
    fn add(self, rhs: T) -> Self::Output {
        Tensor::from(self.a + rhs)
    }
}

impl<T: HasAfEnum + ImplicitPromote<f64, Output = T>> Add<Tensor<T>> for f64 where f64: ImplicitPromote<T> {
    type Output = Tensor<T>;
    fn add(self, rhs: Tensor<T>) -> Self::Output {
        Tensor::from(rhs.a + self)
    }
}

impl<T: HasAfEnum + ImplicitPromote<f64, Output = T>> Div<Tensor<T>> for f64 where f64: ImplicitPromote<T> {
    type Output = Tensor<T>;
    fn div(self, rhs: Tensor<T>) -> Self::Output {
        Tensor::from((self / rhs.a).cast())
    }
}

impl<T: HasAfEnum + Clone + ConstGenerator<OutType = T>> Div<T> for Tensor<T> {
    type Output = Tensor<T>;
    fn div(self, rhs: T) -> Self::Output {
        Tensor::from(self.a / rhs)
    }
}
