extern crate calcu_rs;

use calcu_rs::{base, prelude::*};

fn main() {
    let e1 = base!(1 / 100);
    let e2 = base!(1 / 3).pow(base!(2 / 1000));

    println!("{e1}");
    println!("{e2}");
}
