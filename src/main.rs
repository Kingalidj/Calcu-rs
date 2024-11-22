use std::time::Instant;

use calcu_rs::{expr as e, SymbolicExpr};

fn main() {
    env_logger::builder()
        .filter_level(log::LevelFilter::Info)
        .format_timestamp(None)
        .init();

    let start = Instant::now();

    //let mut a = e!((x^2-1)/(x+1));
    //println!("{}", a);

    // (x^2-1)/(x+1) = x

    // let e = e!(sin(x) + sin(y) - 2*sin(x/2 + y/2)*cos(x/2 - y/2)).reduce();
    //rintln!("{e}");
    //println!("{}", e.simplify_trig().reduce());
    // let e = e.contract_trig().expand();
    // println!("{e}");
    // println!("{}", e.reduce());

     println!("{}", e!(add_raw(-x - y, x)).reduce());
    // println!("{}", e!(mul_raw(sin(x) * sin(y), 1/ sin(x) * 1/ sin(y))).reduce());
    // println!();
    // println!();
    //let n = e.numerator();
    //println!("{n}");

    //println!("{a}");
     let mut arr = vec![e!(x), e!(y), e!(-x), e!(-y)];
     arr.sort();
     println!("{:?}", arr);
     println!("{}", e!(x) > e!(y));
     println!("{}", e!(-x) > e!(-y));
     println!("{}", e!(-x) > e!(y));

    //a.clear_explanation();
    //let eclass = a.simplify();

    //for e in &eclass {
    //    println!("{}: {e}", ExprCost::cost(e));
    //}
    //

    //println!("\nstep-by-step:\n{:?}", eclass[0].clone().steps());

    //let e = e!((x^2 - 1)/(x+1)).expand().rationalize().cancel();
    //println!("{}", e);

    //let a = e!(12);
    //let b = e!(43);
    //let c = a / b;

    //println!("{:?}", c.steps());

    //}

    log::info!("took: {:?}ms", (Instant::now() - start).as_micros());
}
