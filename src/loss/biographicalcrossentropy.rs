use arrayfire::{af_print, Array, clamp, mul, gt, HasAfEnum, ImplicitPromote, log, mean, selectl, sum, lt};

pub struct BiographicalCrossEntropy<T: HasAfEnum> {pub placeholder: Array<T>}
impl<T: HasAfEnum + ImplicitPromote<f32>> BiographicalCrossEntropy<T> where f32: ImplicitPromote<T> {
    pub fn calculate(&self, pred: &Array<T>, target: &Array<u32>) -> f64 {
        return match target.numdims() {
            1 => {
                // Yes, shout at me for doing it like this. ik this is possibly th most inefficient way to transfer a scalar array to the cpu, but it is the best i could think of.
                let mut b: Vec<f64> = Vec::new();
                let c = mean(&-log(&pred).cast::<f64>(), 1);
                b.resize(c.elements(), 0.0);
                c.host(&mut b);
                b[0]}

            _ => {
                let mut b: Vec<f64> = Vec::new();
                let c = mean(&-log(&(pred * &(target.cast::<T>()))).cast::<f64>(), 1);
                b.resize(c.elements(), 0.0);
                c.host(&mut b);
                b[0]
            }
        }
    }
}