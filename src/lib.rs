pub mod base;
pub mod derivative;
pub mod numeric;
pub mod operator;
pub mod pattern;
pub mod rational;

pub mod prelude {
    pub use crate::base::{self, Base, CalcursType, Num, Symbol};
    pub use crate::numeric::{Infinity, Undefined};
    pub use crate::rational::Rational;
}
