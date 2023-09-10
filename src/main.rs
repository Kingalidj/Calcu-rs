extern crate calcu_rs;
use calcu_rs::*;

fn main() {
    println!("{:?}", (Symbol::new("a") & TRUE));
    println!(
        "{:?}",
        (Symbol::new("a") & TRUE)
            .subs("a", FALSE.to_basic())
            .simplify()
    );
}
