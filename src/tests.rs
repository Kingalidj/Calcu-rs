mod test_display {
    use crate::prelude::*;
    use calcu_rs::calc;
    use pretty_assertions::assert_eq;
    use test_case::test_case;

    macro_rules! c {
        ($($x: tt)*) => {
            calc!($($x)*)
        }
    }

    #[test_case(c!(x^(-1)), "1/x")]
    #[test_case(c!(34/3), "34/3")]
    #[test_case(c!(x^(-3)), "x^(-3)")]
    #[test_case(c!(x^2), "x^2")]
    #[test_case(c!(x+x), "2x")]
    #[test_case(c!(1^2), "1")]
    #[test_case(c!((1/2)^2), "1/4")]
    #[test_case(c!((1/3)^(1/100)), "(1/3)^(1/100)")]
    #[test_case(c!((10^15) + 1/1000), "1000000000000000001 e-3")]
    #[test_case(c!((1/3)^(2/1000)), "(1/3)^(1/500)")]
    fn disp_fractions(exp: Base, res: &str) {
        let fmt = format!("{}", exp);
        assert_eq!(fmt, res);
    }
}
mod test_rational {
    use crate::prelude::*;
    use pretty_assertions::assert_eq;

    macro_rules! r {
        ($v: literal) => {
            Rational::new($v, 1)
        };

        ($numer: literal / $denom: literal) => {
            Rational::new($numer, $denom)
        };
    }

    #[test]
    fn exprs() {
        assert_eq!(r!(1) + r!(1), Some(r!(2)));
        assert_eq!(r!(1 / 3) + r!(2 / 3), Some(r!(1)));
        assert_eq!(r!(1 / 3) - r!(2 / 3), Some(r!(-1 / 3)));
        assert_eq!(r!(1 / -3) * r!(3), Some(r!(-1)));
        assert!(r!(2) > r!(1));
        assert!(r!(2) >= r!(2));
        assert!(r!(2 / 4) <= r!(4 / 8));
        assert!(r!(5 / 128) > r!(11 / 2516));
    }
}
mod test_derivative {
    use crate::calc;
    use crate::prelude::*;
    use pretty_assertions::assert_eq;
    use test_case::test_case;

    macro_rules! c {
        ($($x: tt)*) => {
            calc!($($x)*)
        }
    }

    #[test_case(1, c!((x^2) + x*3), c!(2*x + 3))]
    #[test_case(2, c!(1/3 + 3/5),   c!(0))]
    #[test_case(3, c!(x+y),         c!(1))]
    fn sum_rule(_case: u32, f: Base, df: Base) {
        assert_eq!(f.derive("x"), df);
    }

    #[test_case(c!((x^2)*y), c!(2*x*y); "1")]
    fn product_rule(f: Base, df: Base) {
        assert_eq!(f.derive("x"), df);
    }

    #[test_case(c!(x).derive("x"), c!(1))]
    #[test_case(c!(y).derive("x"), c!(0))]
    #[test_case(c!(x*x).derive("x"), c!(2*x))]
    #[test_case(c!((x^2 - x) / (2 * x)).derive("x"), c!(1 / 2))]
    fn derive(expr: Base, result: Base) {
        assert_eq!(expr, result);
    }
}
mod test_operators {
    use crate::calc;
    use crate::prelude::*;
    use pretty_assertions::assert_eq;
    use test_case::test_case;

    macro_rules! c {
        ($($t:tt)*) => {
            calc!($($t)*)
        }
    }

    #[test_case(c!(2 + 3),      c!(5);      "1")]
    #[test_case(c!(1/2 + 1/2),  c!(1);      "2")]
    #[test_case(c!(x + x),      c!(x * 2);  "3")]
    #[test_case(c!(-3 + 1 / 2), c!(-5 / 2); "4")]
    #[test_case(c!(oo + 4),     c!(oo);     "5")]
    #[test_case(c!(-oo + 4),    c!(-oo);    "6")]
    #[test_case(c!(oo + oo),    c!(oo);     "7")]
    #[test_case(c!(-oo + oo),   c!(undef);  "8")]
    #[test_case(c!(undef + oo), c!(undef);  "9")]
    #[test_case(c!(4/2 + 0),    c!(2);      "10")]
    fn add(add: Base, sol: Base) {
        assert_eq!(add, sol);
    }

    #[test_case(c!(-1 - 3),        c!(-4);     "1")]
    #[test_case(c!(-3 - 1 / 2),    c!(-7 / 2); "2")]
    #[test_case(c!(1 / 2 - 1 / 2), c!(0);      "3")]
    #[test_case(c!(oo - 4),        c!(oo);     "4")]
    #[test_case(c!(-oo - 4 / 2),   c!(-oo);    "5")]
    #[test_case(c!(oo - 4),        c!(oo);     "6")]
    #[test_case(c!(oo - oo),       c!(undef);  "7")]
    #[test_case(c!(-oo - oo),      c!(-oo);    "8")]
    #[test_case(c!(undef - oo),    c!(undef);  "9")]
    fn sub(sub: Base, sol: Base) {
        assert_eq!(sub, sol)
    }

    #[test_case(c!(-1*3),         c!(-3);     "1")]
    #[test_case(c!(-1*0),         c!(0);      "2")]
    #[test_case(c!(-1*3) * c!(0), c!(0);      "3")]
    #[test_case(c!(-3*1 / 2),     c!(-3 / 2); "4")]
    #[test_case(c!(1 / 2*1 / 2),  c!(1 / 4);  "5")]
    #[test_case(c!(oo*4),         c!(oo);     "6")]
    #[test_case(c!(-oo * 4/2),    c!(-oo);    "7")]
    #[test_case(c!(oo*4),         c!(oo);     "8")]
    #[test_case(c!(oo*-1),        c!(-oo);    "9")]
    #[test_case(c!(oo*oo),       c!(oo);      "10")]
    #[test_case(c!(-oo*oo),      c!(-oo);     "11")]
    #[test_case(c!(undef*oo),    c!(undef);   "12")]
    fn mul(mul: Base, sol: Base) {
        assert_eq!(mul, sol);
    }

    #[test_case(c!(0/0), c!(undef); "1")]
    #[test_case(c!(0/5), c!(0);     "2")]
    #[test_case(c!(5/0), c!(undef); "3")]
    #[test_case(c!(5/5), c!(1);     "4")]
    #[test_case(c!(1/3), c!(1/3);   "5")]
    #[test_case(c!(x/x), c!(1);     "6")]
    #[test_case(c!((x*x + x) / x), c!(x + 1); "7")]
    #[test_case(c!((x*x + x) / (1 / x)), c!(x); "8")]
    fn div(div: Base, sol: Base) {
        assert_eq!(div, sol);
    }

    #[test_case(c!(1^(1/100)),  c!(1);     "1")]
    #[test_case(c!(4^1),        c!(4);     "2")]
    #[test_case(c!(0^0),        c!(undef); "3")]
    #[test_case(c!(0^(-3/4)),   c!(undef); "4")]
    #[test_case(c!(0^(3/4)),    c!(0);     "5")]
    #[test_case(c!((1/2)^(-1)), c!(4/2);   "6")]
    #[test_case(c!((x^2)^3),    c!(x^6);   "7")]
    fn pow(pow: Base, sol: Base) {
        assert_eq!(pow, sol);
    }

    #[test_case(c!(x*x*2 + 3*x + 4/3), c!(4/3 + (x^2) * 2 + 3*x); "1")]
    fn polynom(p1: Base, p2: Base) {
        assert_eq!(p1, p2);
    }
}
