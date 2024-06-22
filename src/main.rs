use calcu_rs::*;

fn main() {
    std::env::set_var("RUST_LOG", "=info");
    env_logger::init();
    mod_main();

    //let expr = pat!(x + x);

    //let mut lhs = ExprTree::default();
    //let exp = lhs.push(Node::Rational(Rational::from((321u64, 43u64))));
    //let base = lhs.push(Node::Rational(Rational::from(41)));
    //lhs.push(Node::Pow([base, exp]));
    //println!("{}", lhs);
    //lhs.simplify();
    //println!("{}", lhs);
    //lhs.cleanup();
    //println!("{}", lhs);

    //let exp = Rational::from((321u64, 43u64));
    //let base = Rational::from(41);
    //println!("{:?}", base.pow(exp));

    //let expr = calc!(x ^ 2 + 2 * x * y + y ^ 2);
    //let best = GraphExpr::analyse(
    //    &expr,
    //    Duration::from_millis(5000),
    //    &GraphExpr::scalar_rules(),
    //    GraphExprCostFn,
    //);

    //println!("{} => {}", expr, best);
}
