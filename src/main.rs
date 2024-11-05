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

    let mut a = e!((x^2-1)/(x+1));

    // (x^2-1)/(x+1) = x


    println!("{a}");

    a.clear_explanation();
    let eclass = a.simplify();

    for e in &eclass {
        println!("{}: {e}", ExprCost::cost(e));
    }
    

    println!("\nstep-by-step:\n{:?}", eclass[0].clone().steps());

    //let e = e!((x^2 - 1)/(x+1)).expand().rationalize().cancel();
    //println!("{}", e);

    //let a = e!(12);
    //let b = e!(43);
    //let c = a / b;

    //println!("{:?}", c.steps());
    


    //}

    log::info!("took: {:?}ms", (Instant::now() - start).as_micros());
}
