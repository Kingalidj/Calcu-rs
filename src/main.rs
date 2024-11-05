use std::{hash::{DefaultHasher, Hasher}, time::Instant};

use calcu_rs::prelude::*;

use calcu_rs::{expr as e, SymbolicExpr, algos::{ExprCost, CostFn}};

fn calculate_hash<T: std::hash::Hash>(t: &T) -> u64 {
    let mut hasher = DefaultHasher::new();
    t.hash(&mut hasher);
    hasher.finish()
}


fn main() {
    env_logger::builder()
        .filter_level(log::LevelFilter::Info)
        .format_timestamp(None)
        .init();

    let start = Instant::now();

    let a = e!(y^2 * 2*(-y*2 + x));
    println!("{a}");

    let eclass = a.simplify();

    for e in &eclass {
        println!("{}: {e}", ExprCost::cost(&e));
    }
    


    //}

    log::info!("took: {:?}ms", (Instant::now() - start).as_micros());
}
