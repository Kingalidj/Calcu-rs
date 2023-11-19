extern crate calcu_rs;

use calcu_rs::{base, prelude::*};

fn main() {
    println!("{}", base!(pos_inf) > base!(3));
    println!("{}", base!(neg_inf) > base!(3));
    println!("{}", base!(3) >= base!(6 / 2));
    println!("{}", base!(0) ^ base!(-1 / 3));
}
