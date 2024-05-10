extern crate calcu_rs;

use std::time::Duration;
use calcu_rs::e_graph::*;
use calcu_rs::prelude::*;
use egg::{Extractor, RecExpr, Runner};

fn main() {
    std::env::set_var("RUST_LOG", "egg=warn");
    env_logger::init();

    let expr = calc!(2 * a + 3 * a);
    let best = GraphExpr::analyse(&expr, Duration::from_millis(1000), &GraphExpr::basic_rules(), egg::AstSize);

    println!("{} => {}", expr, best.simplify());

}
