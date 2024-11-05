#![allow(dead_code)]

pub extern crate self as calcu_rs;

pub mod algos;
pub mod atom;
mod fmt_ast;
mod polynomial;
mod rational;
mod rubi;
mod transforms;
mod utils;

pub use atom::{Expr, SymbolicExpr};
pub use calcurs_macros::expr;

pub mod prelude {
    pub use crate::atom::{Expr, Irrational, Pow, Prod, Sum, SymbolicExpr};
    pub use crate::rational::Rational;
}
