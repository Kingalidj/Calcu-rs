use std::{collections::VecDeque, fmt, ops};

use derive_more::IsVariant;

use crate::{
    atom::{self, unicode, Atom, Irrational, SymbolicExpr},
    rational::Rational,
};

#[derive(Clone, PartialEq, Eq, Hash, Debug, IsVariant)]
pub enum FmtAtom {
    Undef,
    Rational(Rational),
    Irrational(Irrational),
    Var(String),
    Sum(VecDeque<FmtAtom>),
    Prod(VecDeque<FmtAtom>),
    Pow(Box<FmtAtom>, Box<FmtAtom>),
    Func(atom::Func, Vec<FmtAtom>),
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
                if pow.exponent().atom().is_min_one() {
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
                func.clone(),
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

pub trait SymbolicFormatter {
    /// x [-] y
    fn symbl_sub(f: &mut fmt::Formatter<'_>) -> fmt::Result;
    /// x [+] y
    fn symbl_add(f: &mut fmt::Formatter<'_>) -> fmt::Result;
    /// x [*] y
    fn symbl_mul(f: &mut fmt::Formatter<'_>) -> fmt::Result;
    /// x [/] y
    fn symbl_div(f: &mut fmt::Formatter<'_>) -> fmt::Result;
    /// x[^]y
    fn symbl_pow(f: &mut fmt::Formatter<'_>) -> fmt::Result;
    /// e.g x[ ]+[ ]y
    fn space(f: &mut fmt::Formatter<'_>) -> fmt::Result;
    /// (a[,] b[,] ...)
    fn comma(f: &mut fmt::Formatter<'_>) -> fmt::Result;

    /// [(] ... )
    fn lparen(f: &mut fmt::Formatter<'_>) -> fmt::Result;
    /// ( ... [)]
    fn rparen(f: &mut fmt::Formatter<'_>) -> fmt::Result;

    fn undef(f: &mut fmt::Formatter<'_>) -> fmt::Result;

    fn rational(r: &Rational, f: &mut fmt::Formatter<'_>) -> fmt::Result;

    fn irrational(i: &Irrational, f: &mut fmt::Formatter<'_>) -> fmt::Result;

    fn var(v: &str, f: &mut fmt::Formatter<'_>) -> fmt::Result;

    fn sum(args: &VecDeque<FmtAtom>, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut args = args.iter();

        if let Some(a) = args.next() {
            Self::atom(a, f)?;
        }
        for a in args {
            match &a {
                FmtAtom::UnrySub(e) => {
                    Self::space(f)?;
                    Self::symbl_sub(f)?;
                    Self::space(f)?;
                    Self::fmt_w_prec(sum_prec(), e, f)?;
                }
                _ => {
                    Self::space(f)?;
                    Self::symbl_add(f)?;
                    Self::space(f)?;
                    Self::fmt_w_prec(sum_prec(), a, f)?;
                }
            };
        }

        Ok(())
    }

    fn prod(args: &VecDeque<FmtAtom>, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use FmtAtom as F;
        let mut args = args.iter().peekable();

        let mut prev: Option<&FmtAtom> = None;
        while let Some(curr) = args.next() {
            let next = args.peek();
            match (prev, &curr, next) {
                (Some(a), b, _) if implicit_postfix_mul(a) && implicit_prefix_mul(b) => {
                    Self::fmt_w_prec(prod_prec(), b, f)?;
                }
                (Some(F::Sum(_)), F::Sum(_), _) => {
                    Self::fmt_w_prec(prod_prec(), curr, f)?;
                }
                (Some(F::Func(..)), F::Func(..), _) => {
                    Self::fmt_w_prec(prod_prec(), curr, f)?;
                }
                (Some(_), F::Fraction(n, d), _) => {
                    Self::frac(n, d, f)?;
                }
                (Some(p), curr, _) if p.is_func_pow_number() && curr.is_func_pow_number() => {
                    Self::fmt_w_prec(prod_prec(), curr, f)?;
                }
                (Some(F::Pow(..)), _, _) => {
                    Self::space(f)?;
                    Self::symbl_mul(f)?;
                    Self::space(f)?;
                    Self::fmt_w_prec(prod_prec(), curr, f)?;
                }
                _ => {
                    if prev.is_some() {
                        Self::symbl_mul(f)?;
                    }
                    Self::fmt_w_prec(prod_prec(), curr, f)?;
                }
            };
            prev = Some(curr);
        }

        Ok(())
    }

    fn pow(b: &FmtAtom, e: &FmtAtom, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use FmtAtom as F;
        match (b, e) {
            (F::Func(func, args), e) if e.is_number() => {
                Self::var(&func.name(), f)?;
                Self::symbl_pow(f)?;
                Self::atom(e, f)?;
                return Self::func_args(args, f)
            }
            _ => (),
        }
        Self::fmt_w_prec(pow_prec() + 1, b, f)?;
        Self::symbl_pow(f)?;
        Self::fmt_w_prec(pow_prec() + 1, e, f)
    }

    /// [n/d]
    #[inline(always)]
    fn frac(n: &FmtAtom, d: &FmtAtom, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        Self::fmt_w_prec(prod_prec(), n, f)?;
        Self::symbl_div(f)?;
        Self::fmt_w_prec(pow_prec(), d, f)
    }

    /// [-x]
    #[inline(always)]
    fn unry_sub(x: &FmtAtom, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        Self::symbl_sub(f)?;
        Self::atom(x, f)
    }

    fn func_args(args: &[FmtAtom], f: &mut fmt::Formatter<'_>) -> fmt::Result {
        Self::lparen(f)?;
        let mut args = args.iter();
        if let Some(a) = args.next() {
            Self::atom(a, f)?;
        }
        for a in args {
            Self::comma(f)?;
            Self::space(f)?;
            Self::atom(a, f)?;
        }
        Self::rparen(f)
    }

    #[inline(always)]
    fn func(func: &atom::Func, args: &[FmtAtom], f: &mut fmt::Formatter<'_>) -> fmt::Result {
        Self::var(&func.name(), f)?;
        Self::func_args(args, f)
    }

    fn fmt_w_prec(prec: u32, e: &FmtAtom, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if prec <= e.prec() {
            Self::atom(e, f)
        } else {
            Self::lparen(f)?;
            Self::atom(e, f)?;
            Self::rparen(f)
        }
    }

    fn atom(a: &FmtAtom, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use FmtAtom as FA;
        match a {
            FA::Undef => Self::undef(f),
            FA::Rational(r) => Self::rational(r, f),
            FA::Irrational(i) => Self::irrational(i, f),
            FA::Var(v) => Self::var(v, f),
            FA::Sum(sum) => Self::sum(sum, f),
            FA::Prod(prod) => Self::prod(prod, f),
            FA::Pow(b, e) => Self::pow(b, e, f),
            FA::Func(func, args) => Self::func(func, args, f),
            FA::Fraction(n, d) => Self::frac(n, d, f),
            FA::UnrySub(x) => Self::unry_sub(x, f),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct UnicodeFmt;

impl SymbolicFormatter for UnicodeFmt {
    #[inline]
    fn symbl_sub(f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", unicode::sub())
    }

    #[inline]
    fn symbl_add(f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", unicode::add())
    }

    #[inline]
    fn symbl_mul(f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", unicode::mul())
    }

    #[inline]
    fn symbl_div(f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", unicode::frac_slash())
    }

    #[inline]
    fn symbl_pow(f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", unicode::pow())
    }

    #[inline]
    fn space(f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, " ")
    }

    #[inline]
    fn comma(f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, ",")
    }

    #[inline]
    fn lparen(f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "(")
    }

    #[inline]
    fn rparen(f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, ")")
    }

    #[inline]
    fn undef(f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", unicode::undef())
    }

    #[inline]
    fn rational(r: &Rational, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{r}")
    }

    #[inline]
    fn irrational(i: &Irrational, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{i}")
    }

    #[inline]
    fn var(v: &str, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{v}")
    }

    #[inline]
    fn unry_sub(x: &FmtAtom, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}{x}", unicode::unry_sub())
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

    fn is_number(&self) -> bool {
        self.is_rational() || self.is_irrational()
    }
    fn is_func_pow_number(&self) -> bool {
        match self {
            FmtAtom::Func(..) => true,
            FmtAtom::Pow(b, e) if b.is_func() && e.is_number() => true,
            _ => false,
        }
    }
}

impl fmt::Display for FmtAtom {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        UnicodeFmt::atom(self, f)
    }
}

fn implicit_prefix_mul(e: &FmtAtom) -> bool {
    use FmtAtom as F;
    match e {
        F::Var(_)
        | F::Rational(_)
        | F::Irrational(_)
        | F::Sum(_)
        | F::Func(_, _)
        | F::Pow(_, _) => true,
        _ => false,
    }
}

fn implicit_postfix_mul(e: &FmtAtom) -> bool {
    use FmtAtom as F;
    match e {
        F::Rational(_) | F::Irrational(_) => true,
        _ => false,
    }
}

#[cfg(test)]
mod test_unicode_fmt {
    use super::atom::Expr;

    use calcurs_macros::expr as e;

    #[test]
    fn basic() {
        let fmt_res = vec![
            (e!(a + b), "a + b"),
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
            (e!(sin(x) ^ 3 * tan(x)), "sin^3(x)tan(x)"),
            (e!(sin(x) ^ (x + y) * tan(x)), "sin(x)^(x + y) · tan(x)"),
            (e!(sin(x) ^ x * tan(x)), "sin(x)^x · tan(x)"),
            (e!(3 * sin(x) ^ pi * cos(x) ^ 3), "3sin^π(x)cos^3(x)"),
            (e!(sin(x) * sin(x)), "sin^2(x)"),
            (e!(x ^ y ^ z), "(x^y)^z"),
            (e!(x ^ (y ^ z)), "x^(y^z)"),
        ];

        for (e, res) in fmt_res {
            assert_eq!(e.to_string(), res)
        }
    }
}
