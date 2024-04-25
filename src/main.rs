extern crate calcu_rs;

use calcu_rs::prelude::*;

fn main() {
    let e = calc!((x * x + x) / (x));
    let e = calc!((x * x + x) / (1 / x));
    println!("{}", e);
}
