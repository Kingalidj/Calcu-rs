extern crate calcu_rs;

use calcu_rs::prelude::*;

fn main() {
    let x = Variable::new("x").to_basic();
    let y = Variable::new("y").to_basic();
    let z = Variable::new("z").to_basic();

    let expr = (x.clone() & y.clone()) | z.clone();
    println!("{}", expr);

    let expr = expr.subs("x", y).subs("y", True).subs("z", False);
    println!("{}", expr.simplify());
}
