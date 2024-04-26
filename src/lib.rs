pub extern crate self as calcu_rs;
pub use calcurs_macros::{calc, identity};

pub mod base;
pub mod derivative;
pub mod numeric;
pub mod operator;
pub mod pattern;
pub mod rational;

pub mod prelude {
    pub use crate::{
        base::{self, Base, CalcursType, Symbol},
        calc,
        derivative::Differentiable,
        numeric::{Float, Infinity, Numeric, Sign},
        operator::{Add, Div, Mul, Pow, Sub},
        rational::Rational,
    };
}

#[cfg(test)]
mod tests;
