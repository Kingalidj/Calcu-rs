extern crate calcu_rs;

use calcu_rs::prelude::*;

use num::BigRational;

fn main() {
    let x = Variable::new("x");
    let i = Integer::new(3);
    let res = Add::add(x.clone(), i.clone());
    println!("{}", Add::add(res, x));
}
