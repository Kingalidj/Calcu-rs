extern crate calcu_rs;
use calcu_rs::*;

fn main() {
    std::env::set_var("RUST_BACKTRACE", "1");

    println!("{:?}", (TRUE & FALSE) & Symbol::new("x"));
}
