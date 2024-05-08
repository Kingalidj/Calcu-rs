extern crate calcu_rs;

use egg::{CostFunction, Extractor, Id, Language, RecExpr, Runner};
use calcu_rs::e_graph::*;
use calcu_rs::prelude::*;

fn main() {
    std::env::set_var("RUST_LOG", "egg=warn");
    env_logger::init();
    let eexpr: RecExpr<EggExpr> = EggExpr::build(&calc!(a * 2 + b * a)).unwrap();
    let runner = Runner::default().with_expr(&eexpr).run(&EggExpr::make_rules());
    let extractor = Extractor::new(&runner.egraph, egg::AstSize);
    let (bc, be) = extractor.find_best(runner.roots[0]);
    println!("{} => {}", eexpr, be);
}