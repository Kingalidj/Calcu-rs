extern crate calcu_rs;

use calcu_rs::{base, prelude::*};


use std::time::{Duration, Instant};

fn main() {
    //let e = (base!(v: x).pow(base!(2)) + base!(1)) / base!(v:x);
    //println!("{}", e);
    //let de = e.derive("x");
    //println!("{}", de);

    {
        let mut n = base!(f: 0);
        let one = base!(f: 1);

        let start = Instant::now();
        for _ in 0..100000000 {
           n += one.clone(); 
        }
        let duration = start.elapsed();
        println!("calcu-rs: {:?}: {n}", duration);
    }

}
