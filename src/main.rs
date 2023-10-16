extern crate calcu_rs;

use calcu_rs::prelude::*;

fn main() {
    let i1 = Integer::new(3).base();
    let i2 = Integer::new(2).base();
    println!("{}", i1 * i2);
}
