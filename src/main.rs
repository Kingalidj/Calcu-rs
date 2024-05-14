extern crate calcu_rs;

use calcu_rs::e_graph::*;
use calcu_rs::prelude::*;
use std::time::Duration;
use calcu_rs::pattern::Item;

fn main() {

    std::env::set_var("RUST_LOG", "egg=warn");
    env_logger::init();

    let expr = calc!(0/0);
    let best = GraphExpr::analyse(
        &expr,
        Duration::from_millis(100),
        &GraphExpr::scalar_rules(),
        GraphExprCostFn,
    );

    println!("{} => {}", expr, best);
}
