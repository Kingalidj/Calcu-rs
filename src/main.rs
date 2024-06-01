extern crate calcu_rs;

use calcu_rs::*;

fn main() {
    std::env::set_var("RUST_LOG", "egg=warn");
    env_logger::init();

    println!("Hello World!");
    

    let graph = expr!(x + 2 * (3 + 4) * 5);
    println!("{:?}", graph);

    //let expr = calc!(x ^ 2 + 2 * x * y + y ^ 2);
    //let best = GraphExpr::analyse(
    //    &expr,
    //    Duration::from_millis(5000),
    //    &GraphExpr::scalar_rules(),
    //    GraphExprCostFn,
    //);

    //println!("{} => {}", expr, best);
}
