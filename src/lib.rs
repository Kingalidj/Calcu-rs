extern crate self as calcu_rs;

#[allow(dead_code)]
mod egraph;

mod expression;
mod rational;
mod rules;
mod utils;

/// Provides datastructures to help with formatting expressions
///
mod fmt_ast;

pub use calcurs_macros::{define_rules, expr, pat};

#[allow(unused_imports)]
pub(crate) use crate::utils::*;

pub use crate::{
    expression::{Expr, ExprContext, Node, ID},
    rational::Rational,
    rules::*,
};

use std::cell::{Ref, RefCell};

pub fn mod_main() {
    env_logger::builder()
        .filter_level(log::LevelFilter::Warn)
        .format_timestamp(None)
        .init();

    let c = ExprContext::new();
    let e1 = expr!(c: (x^2 + x*y + y*x + y^2)^2);
    let e1 = e1.apply_rules(ExprFold, &scalar_rules());
    let fmt = c.fmt_id(e1.id());
    println!("{}", fmt);
    c.to_dot_to_png("expr_context.png").unwrap()
}
