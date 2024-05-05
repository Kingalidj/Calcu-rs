extern crate calcu_rs;

use calcu_rs::prelude::*;

fn main() {
    let a = calc!(x + y + y).simplify();
    println!("{}", a);
}
