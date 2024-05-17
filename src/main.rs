extern crate calcu_rs;

use calcu_rs::e_graph::*;
use calcu_rs::prelude::*;
use std::time::Duration;

pub use egg::*;



fn main() {
    //let p = calc!(y + x);
    //let q = calc!(z + y);
    //println!("{} - {}", p, q);
    //let diff = (p - q).simplify();
    //println!("{}", diff);

    std::env::set_var("RUST_LOG", "egg=warn");
    env_logger::init();


    ////TODO preserve structure when using calc?;
    //let expr = calc_raw!(a * -1);
    //println!("{:?}", expr);
    //GraphExpr::scalar_rules();

    let expr = calc_raw!(1 / 1 / 1);
    let best = expr.simplify();

    println!("{} => {}", expr, best);
}
