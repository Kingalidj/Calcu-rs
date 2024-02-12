extern crate calcu_rs;

use calcu_rs::{base, prelude::*};

fn main() {
    //let e = (base!(v: x).pow(base!(2)) + base!(2)) / base!(v: x);
    let e = (base!(v: x).pow(base!(2)) + base!(1)) / base!(v:x);
    println!("{}", e);
    // println!("{:?}", e);
    let de = e.derive("x");
    println!("{}", de);
    // println!("{:?}", de);
}
