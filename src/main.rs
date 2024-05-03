extern crate calcu_rs;

use calcu_rs::prelude::*;

fn main() {
    let a = calc!(x + y);
    println!("{}", a.is_polynomial_in(&[calc!(w), calc!(y)]));
}
