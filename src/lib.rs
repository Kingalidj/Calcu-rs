extern crate self as calcu_rs;

mod egraph;
mod expression;
mod rational;
mod util;
pub use calcurs_macros::{calc, define_rules, expr};

#[allow(unused_imports)]
pub(crate) use crate::{
    util::*,
};

pub use crate::{
    expression::{ExprGraph, Node, ID},
    rational::Rational,
};

//#[cfg(test)]
//mod tests;
