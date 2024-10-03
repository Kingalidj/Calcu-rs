#![allow(dead_code)]

mod atom;
mod expr;
mod fmt_ast;
mod polynomial;
mod rational;
mod rubi;
mod utils;

pub mod prelude {
    use super::*;
    pub use atom::{Expr, Irrational, Pow, Prod, Sum};
    pub use calcurs_macros::expr;
    pub use rational::Rational;
}
