extern crate calcu_rs;

use calcu_rs::{base, prelude::*};

fn main() {
    let r1 = Rational::new(100000003, 1);
    let r2 = Rational::new(1, 1);

    let e = r1 + r2;
    println!("{e}");
}

