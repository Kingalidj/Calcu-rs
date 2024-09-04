use calcu_rs::prelude::*;

use calcurs_macros::expr as e;

fn main() {
    env_logger::builder()
        .filter_level(log::LevelFilter::Trace)
        .init();
    println!("{}", e!(a * x^2 + b * x + c));

}
