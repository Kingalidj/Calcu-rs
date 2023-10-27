extern crate calcu_rs;

use calcu_rs::prelude::*;

fn main() {
    println!(
        "{}",
        (Rational::int_num(1).base() / Rational::int_num(4).base()) / Rational::int_num(4).base()
    );
}
