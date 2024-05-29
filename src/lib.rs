pub use calcurs_macros::{calc, define_rules, identity};

extern crate self as calcu_rs;
pub mod e_graph;
pub mod egraph;
pub mod expression;
pub mod operator;
pub mod pattern;
pub mod rational;
pub mod scalar;
pub mod util;

pub mod prelude {
    pub use calcu_rs::{
        calc,
        expression::{self, CalcursType, Construct, Expr, Symbol},
        operator::{Diff, Pow, Prod, Quot, Sum},
        rational::Rational,
        scalar::{Float, Infinity, Sign},
    };
}

#[cfg(test)]
mod tests;
