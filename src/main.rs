extern crate calcu_rs;
use calcu_rs::*;

fn main() {
    println!("{:?}", (TRUE & FALSE) & Symbol::new("x"));
}
