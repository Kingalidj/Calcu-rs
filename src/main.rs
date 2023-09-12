extern crate calcu_rs;
use calcu_rs::*;

fn main() {
    std::env::set_var("RUST_BACKTRACE", "1");

    let x = Symbol::typ("x");
    let y = Symbol::typ("y");
    let expr = y & x;
    let expr = expr.subs("x", FALSE);
    println!("{}", expr);
    println!("{}", expr.subs("y", TRUE));
}
