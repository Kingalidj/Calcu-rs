extern crate self as calcu_rs;

#[allow(dead_code)]
mod egraph;

mod expression;
mod rational;
mod util;
pub use calcurs_macros::{calc, define_rules, expr, pat};

#[allow(unused_imports)]
pub(crate) use crate::util::*;

pub use crate::{
    expression::{ExprTree, Node, ID},
    rational::Rational,
};

//#[cfg(test)]
//mod tests;
