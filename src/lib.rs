extern crate core;
pub extern crate self as calcu_rs;

pub use calcurs_macros::{calc, identity};

pub mod expression;
pub mod derivative;
pub mod numeric;
pub mod operator2;
pub use operator2 as operator;
pub mod pattern;
pub mod rational;
pub mod e_graph;

pub mod prelude {
    pub use crate::{
        expression::{self, Expr, CalcursType, Construct, Symbol},
        calc,
        numeric::{Float, Infinity, Sign},
        operator::{Sum, Quot, Prod, Pow, Diff},
        rational::Rational,
    };
}

#[cfg(test)]
mod tests;
