use std::time::Instant;

//use calcu_rs::prelude::*;

use calcu_rs::{expr as e, SymbolicExpr};

fn main() {
    env_logger::builder()
        .filter_level(log::LevelFilter::Info)
        .format_timestamp(None)
        .init();

    let start = Instant::now();
    let e = e!((x*y)^2);
    println!("{}", e);
    println!("{}", e.reduce());

    log::info!("took: {:?}ms", (Instant::now() - start).as_micros());
}
