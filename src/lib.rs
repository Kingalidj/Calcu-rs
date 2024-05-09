extern crate core;
pub extern crate self as calcu_rs;

pub use calcurs_macros::{calc, identity};

pub mod derivative;
pub mod expression;
pub mod numeric;
pub mod operator2;
pub use operator2 as operator;
pub mod e_graph;
pub mod pattern;
pub mod rational;

pub mod prelude {
    pub use crate::{
        calc,
        expression::{self, CalcursType, Construct, Expr, Symbol},
        numeric::{Float, Infinity, Sign},
        operator::{Diff, Pow, Prod, Quot, Sum},
        rational::Rational,
    };
}

#[cfg(test)]
mod tests;
