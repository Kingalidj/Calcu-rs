extern crate self as calcu_rs;

#[allow(dead_code)]
mod egraph;

mod expression;
mod rational;
mod rules;
mod utils;
pub use calcurs_macros::{define_rules, expr, pat};

#[allow(unused_imports)]
pub(crate) use crate::utils::*;

pub use crate::{
    expression::{Expr, Node, ID},
    rational::Rational,
    rules::*,
};

pub fn mod_main() {
    //let mut expr = expr!(x ^ 2 + 2 * x * y + y ^ 2);
    //let mut expr = expr!(2^(8/3));
    //let res = expr.apply_rules(&scalar_rules());
    //let res = Rational::from(2).pow(Rational::from((8u64, 3u64)));
    let mut expr = expr!(2^8);
    let res = expr.apply_rules(&scalar_rules());
    println!("\nres: {:?}", res);
}

//#[cfg(test)]
//mod tests;
