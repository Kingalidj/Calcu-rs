extern crate calcu_rs;

use calcu_rs::prelude::*;

fn main() {
    let e = calc!(((x ^ 2) + 1) / x);
    println!("{}", e);
    let de = e.derive("x");
    println!("{}", de);

    println!("{}", calc!(((x + 1) ^ 2)));
}
