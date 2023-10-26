extern crate calcu_rs;

use calcu_rs::prelude::*;

fn main() {
    println!("{}", Rational::int(-3).base() * Rational::frac(1, 2).base());
}
