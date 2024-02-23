extern crate calcu_rs;

use calcu_rs::{base, prelude::*};

fn main() {
    //let e = (base!(v: x).pow(base!(2)) + base!(1)) / base!(v:x);
    //println!("{}", e);
    //let de = e.derive("x");
    //println!("{}", de);

    let e = base!(v:x).pow(base!(0));

    println!("{e}");
}
