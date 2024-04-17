extern crate calcu_rs;

use calcu_rs::prelude::*;

fn main() {
    let e = calc!((x*x + x) / (x));
    println!("{}", e);
}
