use crate::rational::Rational;
use std::{collections::VecDeque, fmt, ops};

use crate::atom::{self, unicode, Atom, Irrational, SymbolicExpr};

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum FmtAtom {
    Undef,
    Rational(Rational),
    Irrational(Irrational),
    Var(String),
    Sum(VecDeque<FmtAtom>),
    Prod(VecDeque<FmtAtom>),
    Pow(Box<FmtAtom>, Box<FmtAtom>),
    Func(String, Vec<FmtAtom>),
    Fraction(Box<FmtAtom>, Box<FmtAtom>),
    UnrySub(Box<FmtAtom>),
}

impl FmtAtom {
    const ONE: FmtAtom = FmtAtom::Rational(Rational::ONE);
    const ZERO: FmtAtom = FmtAtom::Rational(Rational::ZERO);
    const MINUS_ONE: FmtAtom = FmtAtom::Rational(Rational::MINUS_ONE);
}

impl From<&Atom> for FmtAtom {
    fn from(value: &Atom) -> Self {
        match value {
            Atom::Undef => FmtAtom::Undef,
            Atom::Rational(r) => {
                if r.is_neg() {
                    FmtAtom::UnrySub(FmtAtom::Rational(r.clone().abs()).into())
                } else if r.is_fraction() && r.numer().is_one() {
                    let n = Atom::from(r.numer());
                    let d = Atom::from(r.denom());
                    FmtAtom::Fraction(Self::from(&n).into(), Self::from(&d).into())
                } else {
                    FmtAtom::Rational(r.clone())
                }
            }
            Atom::Irrational(i) => FmtAtom::Irrational(*i),
            Atom::Var(v) => FmtAtom::Var(v.to_string()),
            Atom::Prod(atom::Prod { args }) => args
                .iter()
                .map(|a| FmtAtom::from(a.atom()))
                .fold(FmtAtom::ONE, |prod, r| prod * r),
            Atom::Sum(atom::Sum { args }) => {
                //FmtAtom::Sum(args.into_iter().map(|a| FmtAtom::from(a.get())).collect())
                args.iter()
                    .map(|a| FmtAtom::from(a.atom()))
                    .fold(FmtAtom::ZERO, |prod, r| prod + r)
            }
            Atom::Pow(pow) => {
                if pow.exponent().atom() == &Atom::MINUS_ONE {
                    let n = Atom::ONE;
                    FmtAtom::Fraction(Self::from(&n).into(), Self::from(pow.base().atom()).into())
                } else {
                    FmtAtom::Pow(
                        FmtAtom::from(pow.base().atom()).into(),
                        FmtAtom::from(pow.exponent().atom()).into(),
                    )
                }
            }
            Atom::Func(func) => FmtAtom::Func(
                func.name(),
                func.iter_args().map(|a| FmtAtom::from(a.atom())).collect(),
            ),
        }
    }
}
impl ops::Add for FmtAtom {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        use FmtAtom as F;
        match (self, rhs) {
            (F::ZERO, e) | (e, F::ZERO) => e,
            (lhs, F::Sum(mut s)) => {
                s.push_front(lhs);
                F::Sum(s)
            }
            (F::Sum(mut s), rhs) => {
                s.push_back(rhs);
                F::Sum(s)
            }
            (lhs, rhs) => F::Sum([lhs, rhs].into_iter().collect()),
        }
    }
}
impl ops::Sub for FmtAtom {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        let min_one = FmtAtom::Rational(Rational::MINUS_ONE);
        let min_rhs = min_one * rhs;
        self + min_rhs
    }
}
impl ops::Mul for FmtAtom {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        use FmtAtom as F;
        match (self, rhs) {
            (F::Rational(Rational::ONE), e) | (e, F::Rational(Rational::ONE)) => e,
            (F::UnrySub(lhs), F::UnrySub(rhs)) => *lhs * *rhs,
            (F::UnrySub(lhs), rhs) => F::UnrySub((*lhs * rhs).into()),
            (lhs, F::UnrySub(rhs)) => F::UnrySub((lhs * *rhs).into()),
            // TODO use len for deciding if we combine
            (F::Fraction(n, d), e) | (e, F::Fraction(n, d)) => F::Fraction((*n * e).into(), d),
            (lhs, F::Prod(mut s)) => {
                s.push_front(lhs);
                F::Prod(s)
            }
            (F::Prod(mut s), rhs) => {
                s.push_back(rhs);
                F::Prod(s)
            }
            (lhs, rhs) => F::Prod([lhs, rhs].into_iter().collect()),
        }
    }
}

#[allow(clippy::suspicious_arithmetic_impl)]
impl ops::Div for FmtAtom {
    type Output = Self;
    fn div(self, rhs: Self) -> Self::Output {
        let min_one = FmtAtom::Rational(Rational::MINUS_ONE);
        let one_div_rhs = rhs.pow(min_one);
        self * one_div_rhs
    }
}
impl FmtAtom {
    pub fn pow(self, exp: Self) -> Self {
        FmtAtom::Pow(self.into(), exp.into())
    }
}

const fn sum_prec() -> u32 {
    1
}
const fn prod_prec() -> u32 {
    2
}
const fn pow_prec() -> u32 {
    3
}
const fn atom_prec() -> u32 {
    4
}

impl FmtAtom {
    fn prec(&self) -> u32 {
        match self {
            FmtAtom::Func(_, _) | FmtAtom::Undef | FmtAtom::Irrational(_) | FmtAtom::Var(_) => {
                atom_prec()
            }
            FmtAtom::Rational(r) if r.is_int() => atom_prec(),

            FmtAtom::Pow(_, _) => pow_prec(),

            FmtAtom::Prod(_) | FmtAtom::UnrySub(_) => prod_prec(),
            FmtAtom::Fraction(_, _) | FmtAtom::Rational(_) => prod_prec(),

            FmtAtom::Sum(_) => sum_prec(),
        }
    }

    fn fmt_w_prec(prec_in: u32, e: &Self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if prec_in <= e.prec() {
            write!(f, "{e}")
        } else {
            write!(f, "({e})")
        }
    }

    pub fn fmt_sum(args: &VecDeque<FmtAtom>, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut args = args.iter();

        if let Some(a) = args.next() {
            write!(f, "{a}")?;
        }
        for a in args {
            match &a {
                FmtAtom::UnrySub(e) => {
                    write!(f, " {} ", unicode::sub())?;
                    Self::fmt_w_prec(sum_prec(), e, f)?;
                }
                _ => {
                    write!(f, " {} ", unicode::add())?;
                    Self::fmt_w_prec(sum_prec(), a, f)?;
                }
            };
        }

        Ok(())
    }

    pub fn fmt_prod(args: &VecDeque<FmtAtom>, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use FmtAtom as F;
        let mut args = args.iter().peekable();

        let mut prev: Option<&Self> = None;
        while let Some(a) = args.next() {
            let next = args.peek();
            match (prev, &a, next) {
                (Some(F::Sum(_)), F::Sum(_), _)
                | (
                    Some(F::Rational(_)),
                    F::Var(_) | F::Irrational(_) | F::Sum(_) | F::Func(_, _) | F::Pow(_, _),
                    _,
                ) => {
                    Self::fmt_w_prec(prod_prec(), a, f)?;
                    prev = Some(a);
                }
                (Some(F::Func(..)), F::Func(..), _) => {
                    Self::fmt_w_prec(prod_prec(), a, f)?;
                    prev = Some(a)
                }
                (Some(_), F::Fraction(n, d), _) => {
                    Self::fmt_frac(n, d, f)?;
                    prev = Some(a);
                }
                //(_, a, Some(F::Inverse(_))) => {
                //    Self::fmt_w_prec(pow_prec(), a, f)?;
                //    prev = Some(a);
                //}
                _ => {
                    if prev.is_some() {
                        write!(f, "{}", unicode::mul())?;
                    }
                    Self::fmt_w_prec(prod_prec(), a, f)?;
                    prev = Some(a);
                }
            };
        }

        Ok(())
    }

    pub fn fmt_frac(n: &FmtAtom, d: &FmtAtom, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        Self::fmt_w_prec(prod_prec(), n, f)?;
        write!(f, "{}", unicode::frac_slash())?;
        Self::fmt_w_prec(pow_prec(), d, f)
    }

    pub fn fmt_pow(b: &FmtAtom, e: &FmtAtom, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if pow_prec() < b.prec() {
            write!(f, "{b}")
        } else {
            write!(f, "({b})")
        }?;
        write!(f, "{}", unicode::pow())?;
        if pow_prec() < e.prec() {
            write!(f, "{e}")
        } else {
            write!(f, "({e})")
        }
    }

    pub fn fmt_func(name: &String, args: &[Self], f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{name}(")?;
        let mut args = args.iter();
        if let Some(a) = args.next() {
            write!(f, "{a}")?;
        }
        for a in args {
            write!(f, ", {a}")?;
        }
        write!(f, ")")
    }
}

impl fmt::Display for FmtAtom {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FmtAtom::Undef => write!(f, "{}", unicode::undef()),
            FmtAtom::Rational(r) => write!(f, "{r}"),
            FmtAtom::Irrational(i) => write!(f, "{i}"),
            FmtAtom::Var(v) => write!(f, "{v}"),
            FmtAtom::Sum(args) => FmtAtom::fmt_sum(args, f),
            FmtAtom::Prod(args) => FmtAtom::fmt_prod(args, f),
            FmtAtom::Pow(b, e) => FmtAtom::fmt_pow(b, e, f),
            FmtAtom::Func(n, args) => FmtAtom::fmt_func(n, args, f),
            FmtAtom::UnrySub(e) => write!(f, "{}{e}", unicode::unry_sub()),
            FmtAtom::Fraction(n, d) => FmtAtom::fmt_frac(n, d, f),
        }
    }
}

#[cfg(test)]
mod test_unicode_fmt {
    use super::atom::Expr;

    use calcurs_macros::expr as e;

    #[test]
    fn basic() {
        let fmt_res = vec![
            (e!(a + b * c), "a + b·c"),
            (e!(2 * x * y), "2x·y"),
            (e!(2 * x ^ 3), "2x^3"),
            (e!(2 * x ^ (a + b)), "2x^(a + b)"),
            (e!(2 * x ^ (2 * a)), "2x^(2a)"),
            (e!(a / b), "a/b"),
            (Expr::div_raw(e!(a * b), e!(a * b)), "a·b/(a·b)"),
            (e!((x + y) / (x * y)), "(x + y)/(x·y)"),
            (e!((x + y) * (a + b)), "(x + y)(a + b)"),
            (e!(3 * (a + b)), "3(a + b)"),
            (e!(x * (a + b)), "x·(a + b)"),
            (e!((a + b) * x), "(a + b)·x"),
            (e!(x ^ (a + b)), "x^(a + b)"),
            (e!(y + -x), "y − x"),
            (e!(1 / x), "1/x"),
            (e!(y * 1 / x), "y/x"),
            (e!(3 * 1 / x), "3/x"),
            (e!((1 + x) ^ 2), "(1 + x)^2"),
            (e!(2 * pi), "2π"),
            (e!(3 + 1 / 6 * pi), "3 + π/6"),
        ];

        for (e, res) in fmt_res {
            assert_eq!(e.to_string(), res)
        }
    }
}
