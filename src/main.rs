extern crate calcu_rs;
use calcu_rs::*;

fn main() {
    let b = BooleanAtom {};
    let a: Box<dyn CalcrsType> = Box::new(b);
}
