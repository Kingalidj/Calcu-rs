use std::time::Instant;

use calcu_rs::prelude::*;

use calcurs_macros::expr as e;

fn main() {
    env_logger::builder()
        .filter_level(log::LevelFilter::Trace)
        .init();

    let start = Instant::now();
    //println!("{:?}", e!((2*x)^2).as_monomial(&[e!(x)].into()).coeff());
    //println!("{:?}", e!(a / (b*c + b*d)).as_polynomial(&[e!(1/b)].into()).coeffs());
    //println!("{:?}", e!(1/x).as_monomial(&[e!(x)].into()).coeff());
    //println!("{:?}", Rational::from((5u64, 2u64)).div_rem());
    //println!("{}", e!((x+1)^(5/2)).expand().reduce());
    let e = e!(a/x + b/x).rationalize().cancel().expand().cancel();
    println!("{}", e);

    println!("took: {:?}ms", (Instant::now() - start).as_micros());
}
