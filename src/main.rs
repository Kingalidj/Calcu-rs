extern crate calcu_rs;

use calcu_rs::{base, prelude::*};

fn main() {
    let expr = (base!(1)).pow(base!(1 / 100));
    let res = base!(1 / 100);
    println!("{}", expr);
    println!("{}", res);
}
