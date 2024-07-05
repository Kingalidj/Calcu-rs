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

pub(crate) use calcurs_macros::{define_rules, pat};

#[allow(unused_imports)]
pub(crate) use crate::utils::*;

pub use crate::{
    expression::{Expr, ExprContext, Node, ID},
    fmt_ast::{ExprFormatter, FmtAst, FormatWith},
    rational::Rational,
    rules::{scalar_rules, ExprCost, ExprFold},
    utils::{GlobalSymbol, Symbol, SymbolTable},
};
pub use calcurs_macros::expr;

#[cfg(test)]
fn init_logger() {
    let _ = env_logger::builder()
        .is_test(true)
        .format_timestamp(None)
        .try_init();
}

pub fn mod_main() {
    env_logger::builder()
        .filter_level(log::LevelFilter::Info)
        .format_timestamp(None)
        .init();

    let mut c = ExprContext::new();
    let e1 = expr!(c: ((1/(y+7) - (1 / y)))/(1/y) );
    //let e1 = e1.apply_rules(ExprFold, &scalar_rules());
    println!("{}", e1.fmt_ast());

    #[cfg(not(target_arch = "wasm32"))]
    c.to_dot_to_png("expr_context.png").unwrap()
}
