extern crate calcu_rs;

use calcu_rs::e_graph::*;
use calcu_rs::prelude::*;
use egg::{Extractor, RecExpr, Runner};

fn main() {
    std::env::set_var("RUST_LOG", "egg=warn");
    env_logger::init();
    let eexpr: RecExpr<ExprGraph> = ExprGraph::build(&calc!(0 * (a + b))).unwrap();
    let runner = Runner::default()
        .with_expr(&eexpr)
        .run(&ExprGraph::make_rules());
    let extractor = Extractor::new(&runner.egraph, egg::AstSize);
    let (_bc, be) = extractor.find_best(runner.roots[0]);
    println!("{} => {}", eexpr, be);
}
