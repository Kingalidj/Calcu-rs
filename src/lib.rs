pub extern crate self as calcu_rs;
pub use calcurs_macros::{calc, identity};

pub mod base;
pub mod derivative;
pub mod numeric;
pub mod operator;
pub mod pattern;
pub mod rational;

pub mod prelude {
    pub use crate::base::{self, Base, CalcursType, Differentiable, Symbol};
    pub use crate::calc;
    pub use crate::numeric::{Float, Infinity, Numeric, Sign, Undefined};
    pub use crate::operator::{Add, Div, Mul, Pow, Sub};
    pub use crate::rational::Rational;
}
