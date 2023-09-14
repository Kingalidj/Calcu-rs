extern crate calcu_rs;
use calcu_rs::prelude::*;

fn main() {
    std::env::set_var("RUST_BACKTRACE", "1");

    let expr = TRUE | FALSE & TRUE & FALSE;
    let res = true || false && true && false;
    println!("{} == {}", expr.simplify(), res);
}
