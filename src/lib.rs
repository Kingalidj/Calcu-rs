pub mod base;
pub mod numeric;
pub mod operator;
pub mod rational;

pub mod prelude {
    pub use crate::base::{self, Base, CalcursType, Num, Variable};
    pub use crate::numeric::{Infinity, Undefined};
    pub use crate::rational::Rational;
}
