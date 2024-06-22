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
    expression::{ExprContext, Expr, Node, ID},
    rational::Rational,
    rules::*,
};

//TODO LIST:
// non-global expr context
// thread safe global symbol table

pub fn mod_main() {
    //let mut expr = expr!(x ^ 2 + 2 * x * y + y ^ 2);
    //let mut expr = expr!(2^(8/3));
    //let res = expr.apply_rules(&scalar_rules());
    //let res = Rational::from(2).pow(Rational::from((8u64, 3u64)));
    // TODO: a^e / a^f panics
    let mut c = ExprContext::new();
    let mut e1 = expr!(c: a^2 / a^2);
    let e1 = e1.apply_rules(ExprFold, &scalar_rules());
    println!("{}", e1);
    c.to_dot_to_png("expr_context.png").unwrap()
}

//#[cfg(test)]
//mod tests;
