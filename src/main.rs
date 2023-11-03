extern crate calcu_rs;

use calcu_rs::{base, prelude::*};

fn main() {
    let expr = (base!(v: x) ^ base!(2)) + base!(v: x);
    println!("{}", expr);
}
