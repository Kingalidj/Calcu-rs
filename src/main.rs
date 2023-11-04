extern crate calcu_rs;

use calcu_rs::{base, prelude::*};

fn main() {
    let expr = base!(4 / 3) * (base!(v: x) ^ base!(2)) * base!(2) + base!(3) * base!(v: x);
    println!("{expr}");
}
