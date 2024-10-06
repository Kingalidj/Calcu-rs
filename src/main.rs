use std::time::Instant;

//use calcu_rs::prelude::*;

use calcu_rs::{expr as e, SymbolicExpr};

fn main() {
    env_logger::builder()
        .filter_level(log::LevelFilter::Info)
        .format_timestamp(None)
        .init();

    let start = Instant::now();
    //println!("{:?}", e!((2*x)^2).as_monomial(&[e!(x)].into()).coeff());
    //println!("{:?}", e!(a / (b*c + b*d)).as_polynomial(&[e!(1/b)].into()).coeffs());
    //println!("{:?}", e!(1/x).as_monomial(&[e!(x)].into()).coeff());
    //println!("{:?}", Rational::from((5u64, 2u64)).div_rem());
    //println!("{}", e!((x+1)^(5/2)).expand().reduce());
    //let e = e!((x+y)/(x*y)).rationalize();
    //let e = e!(x * ln(x) * sin(x)).derivative(e!(x));
    //println!("{}", e.reduce());
    //println!("{}", e.reduce().rationalize().expand().cancel());
    //println!("{:?}", e!(y + (3 - 4)*x).expand().reduce().fmt_ast());
    //println!("{}", e!((x+1)*(x-1)).reduce().expand().rationalize().factor_out().reduce());

    //let e = e!(x + -1 * y);
    //let e = e!(x * ln(x)).derivative(e!(x)).rationalize().expand().factor_out();
    //let e = e!(((x+1)^2 + 1)^2);
    let e = e!(exp(2*(x+y)));
    println!("{}", e);
    println!("{}", e.expand_exponential());

    println!("took: {:?}ms", (Instant::now() - start).as_micros());
}
