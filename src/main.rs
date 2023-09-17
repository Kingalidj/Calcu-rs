extern crate calcu_rs;
use calcu_rs::prelude::*;

fn main() {
    std::env::set_var("RUST_BACKTRACE", "1");

    let from = DistributivityRule::<And, Or>::FROM;
    let into = DistributivityRule::<And, Or>::INTO;

    println!("from: {}", from);
    println!("into: {}", into);
}
