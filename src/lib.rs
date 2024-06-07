extern crate self as calcu_rs;

#[allow(dead_code)]
mod egraph;

mod expression;
mod rational;
mod utils;
pub use calcurs_macros::{define_rules, expr, pat};

#[allow(unused_imports)]
pub(crate) use crate::utils::*;

pub use crate::{
    expression::{ExprTree, Node, ID},
    rational::Rational,
};

pub fn mod_main() {

    let mut expr = expr!((a * b) + (a * b));
    expr.cleanup();
    expr.simplify();
    println!("{:?}", expr);
}

//#[cfg(test)]
//mod tests;
