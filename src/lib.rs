pub use calcurs_macros::{calc, calc_raw, define_rules, identity};

extern crate self as calcu_rs;
pub mod derivative;
pub mod expression;
pub mod scalar;
pub mod operator;
pub mod e_graph;
pub mod pattern;
pub mod rational;

pub mod prelude {
    pub use crate::{
        calc, calc_raw,
        expression::{self, CalcursType, Expr, Symbol},
        scalar::{Float, Infinity, Sign},
        operator::{Diff, Pow, Prod, Quot, Sum},
        rational::Rational,
    };
}

#[cfg(test)]
mod tests;
