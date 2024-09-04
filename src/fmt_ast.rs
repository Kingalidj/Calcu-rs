use crate::{expr::Expr, rational::Rational};
use std::{
    collections::VecDeque,
    fmt::{self, Display, Write},
    ops,
};

pub type Var<'a> = &'a str;

/// Specifies formatting for the [crate::Expr]
///
/// The basic Operator traits are implemented for constructing the formatted AST \
/// No calculations are performed, e.g 2 * 3 will be outputted as 2 * 3. \
/// All Primitives are stored as [Ref]
#[derive(Debug)]
pub enum FmtAst<'a> {
    Rational(Rational),
    Var(Var<'a>),
    Undef,

    Sub(Sub<'a>),
    Coeff(Coeff<'a>),
    Frac(Frac<'a>),
    Pow(Pow<'a>),
    Sum(Sum<'a>),
    Prod(Prod<'a>),
    SimplProd(SimplProd<'a>),
}

impl FmtAst<'_> {
    fn is_next_rational(&self) -> bool {
        match self {
            FmtAst::Rational(_) | FmtAst::Coeff(_) => true,
            FmtAst::Pow(Pow(base, _)) if base.is_next_rational() => true,
            FmtAst::Frac(Frac(numer, _)) if numer.is_next_rational() => todo!(),
            _ => false,
        }
    }

    fn is_min_one(&self) -> bool {
        match self {
            FmtAst::Rational(r) if *r == Rational::MINUS_ONE => true,
            _ => false,
        }
    }
    fn is_one(&self) -> bool {
        match self {
            FmtAst::Rational(r) if *r == Rational::ONE => true,
            _ => false,
        }
    }
}

// Holds references to primitive [crate::Node]
//
//#[derive(Debug)]
//pub enum Atom<'a> {
//    //Rational(Ref<'a, Rational>),
//    Rational(Rational),
//    Var(Var<'a>),
//    Undefined,
//}

/// [FmtAst] - [FmtAst]
///
#[derive(Debug)]
pub struct Sub<'a>(Box<FmtAst<'a>>, Box<FmtAst<'a>>);

/// [Rational] * [FmtAst], e.g 3(a + b), -b^2
///
#[derive(Debug)]
pub struct Coeff<'a>(Box<FmtAst<'a>>, Box<FmtAst<'a>>);

impl Coeff<'_> {
    fn check_rational_coeff(&self, f: impl Fn(&Rational) -> bool) -> bool {
        if let FmtAst::Rational(r) = self.0.as_ref() {
            f(r)
        } else {
            false
        }
    }

    fn is_rational(&self) -> bool {
        self.check_rational_coeff(|_| true)
    }

    fn is_neg(&self) -> bool {
        self.check_rational_coeff(Rational::is_neg)
    }
}

/// [FmtAst] / [FmtAst], e.g a/3
///
#[derive(Debug)]
pub struct Frac<'a>(Box<FmtAst<'a>>, Box<FmtAst<'a>>);

/// [FmtAst] ^ [FmtAst]
///
#[derive(Debug)]
pub struct Pow<'a>(Box<FmtAst<'a>>, Box<FmtAst<'a>>);

/// [Rational] * Prod([Var]), e.g: Multi-variable polynomial with coefficient
///
/// n * a * b * c, e.g 3 * a * b * c -> 3abc
#[derive(Debug, Default)]
pub struct SimplProd<'a>(VecDeque<FmtAst<'a>>);

/// Sum([FmtAst]), e.g: 1 + a + 2b
///
#[derive(Debug, Default)]
pub struct Sum<'a>(VecDeque<FmtAst<'a>>);

/// Prod([FmtAst]), e.g: 1 * 2 * 3c
///
#[derive(Debug, Default)]
pub struct Prod<'a>(VecDeque<FmtAst<'a>>);

/// Allows for custom formatting of [FmtAst]
///
pub trait ExprFormatter: Sized {
    type Result;
    // helper function, because the formatter decides where to put parenthesis
    fn fmt_paren(&mut self, e: &impl FormatWith<Self>) -> Self::Result;
}

/// implemented by the structs that can be formatted by [ExprFormatter]
///
pub trait FormatWith<EF: ExprFormatter>: FmtPrecedence {
    fn fmt_with(&self, f: &mut EF) -> EF::Result;
    fn fmt_paren(&self, f: &mut EF) -> EF::Result
    where
        Self: Sized,
    {
        f.fmt_paren(self)
    }

    fn fmt_paren_prec(e: &impl FormatWith<EF>, f: &mut EF) -> EF::Result
    where
        Self: Sized,
    {
        if e.prec_of_val() < Self::prec_of() {
            e.fmt_paren(f)
        } else {
            e.fmt_with(f)
        }
    }
}

macro_rules! fa {
    ($ty:ident($e:expr)) => {
        FmtAst::$ty($ty($e.into()).into())
    };
    ($ty:ident($e1:expr, $e2:expr)) => {
        FmtAst::$ty($ty($e1.into(), $e2.into()))
    };
}

impl<T> FmtPrecedence for Box<T>
where
    T: FmtPrecedence,
{
    fn prec_of() -> u32 {
        T::prec_of()
    }
    fn prec_of_val(&self) -> u32 {
        self.as_ref().prec_of_val()
    }
}
impl<T, EF> FormatWith<EF> for Box<T>
where
    T: FormatWith<EF>,
    EF: ExprFormatter,
{
    fn fmt_with(&self, f: &mut EF) -> EF::Result {
        self.as_ref().fmt_with(f)
    }
}

/// for deciding if [FmtAst] in binary operations are wrapped with parenthesis or not
///
/// e.g (a + b)^2 vs a + b^2
pub trait FmtPrecedence {
    fn prec_of() -> u32;
    fn prec_of_val(&self) -> u32 {
        Self::prec_of()
    }
}
macro_rules! impl_precedence {
    ($ty:ty; $prec:expr) => {
        impl FmtPrecedence for $ty {
            fn prec_of() -> u32 {
                $prec
            }
        }
    };
}
impl_precedence!(Sum<'_>;     1);
impl_precedence!(Sub<'_>;     1);
impl_precedence!(Prod<'_>;    2);
impl_precedence!(Coeff<'_>;   2);
impl_precedence!(Frac<'_>;    2);
impl_precedence!(SimplProd<'_>; 2);
impl_precedence!(Pow<'_>;     3);
impl_precedence!(Var<'_>;     4);

impl FmtPrecedence for Rational {
    fn prec_of() -> u32 {
        4
    }
    fn prec_of_val(&self) -> u32 {
        if !self.is_int() {
            Frac::prec_of()
        } else {
            4
        }
    }
}

impl FmtPrecedence for FmtAst<'_> {
    fn prec_of() -> u32 {
        panic!("FmtAst precedence is defined when created")
    }
    fn prec_of_val(&self) -> u32 {
        use FmtAst as FA;
        match self {
            FA::Sub(x) => x.prec_of_val(),
            FA::Coeff(x) => x.prec_of_val(),
            FA::Frac(x) => x.prec_of_val(),
            FA::Pow(x) => x.prec_of_val(),
            FA::Sum(x) => x.prec_of_val(),
            FA::Prod(x) => x.prec_of_val(),
            FA::SimplProd(x) => x.prec_of_val(),
            FA::Rational(x) => x.prec_of_val(),
            FA::Var(x) => x.prec_of_val(),
            FA::Undef => 4,
        }
    }
}

impl<'a> ops::Add for FmtAst<'a> {
    type Output = FmtAst<'a>;

    fn add(self, rhs: Self) -> Self::Output {
        use FmtAst as FA;
        match (self, rhs) {
            (lhs, FA::Rational(Rational::ZERO)) => lhs,
            (FA::Rational(Rational::ZERO), rhs) => rhs,

            (FA::Sum(mut lhs), FA::Sum(rhs)) => {
                lhs.0.extend(rhs.0);
                FA::Sum(lhs)
            }
            (lhs, FA::Sum(mut rhs)) => {
                rhs.0.push_front(lhs);
                FA::Sum(rhs)
            }
            (FA::Sum(mut lhs), rhs) => {
                lhs.0.push_back(rhs);
                FA::Sum(lhs)
            }
            (lhs, rhs) => fa!(Sum([lhs, rhs])),
        }
    }
}

impl<'a> ops::Sub for FmtAst<'a> {
    type Output = FmtAst<'a>;

    fn sub(self, rhs: Self) -> Self::Output {
        fa!(Sub(self, rhs))
    }
}

impl<'a> ops::Mul for FmtAst<'a> {
    type Output = FmtAst<'a>;

    fn mul(self, rhs: Self) -> Self::Output {
        use FmtAst as FA;
        match (self, rhs) {
            (lhs, FA::Rational(Rational::ONE)) => lhs,
            (FA::Rational(Rational::ONE), rhs) => rhs,

            (FA::Coeff(c), rhs) if c.is_rational() => {
                FA::Coeff(Coeff(c.0, (*c.1 * rhs).into()))
            }
            (FA::Coeff(c), rhs) if c.is_rational() => {
                FA::Coeff(Coeff(c.0, (*c.1 * rhs).into()))
            }

            // var1 * var2
            (v1 @ FA::Var(_), v2 @ FA::Var(_)) => fa!(SimplProd([v1, v2])),
            // coeff(c, e1) * e2 -> coeff(c, e1 * e2)
            (FA::Coeff(Coeff(coeff, expr)), rhs) => {
                let mut lhs = expr;
                *lhs = *lhs * rhs;
                fa!(Coeff(coeff, lhs))
            }
            // (a + b) * x => (a + b)x
            (e @ FA::Sum(_), v) /*if var.len() == 1*/ => {
                fa!(Coeff(e, v))
            }
            (v, e @ FA::Sum(_)) /*if var.len() == 1*/ => {
                fa!(Coeff(v, e))
            }
            // e1 * coeff(c, e2) -> coeff(c, e1 * e2)
            //(lhs, FA::Coeff(Coeff(coeff, expr))) => {
            //    let mut rhs = expr;
            //    *rhs = lhs * *rhs;
            //    e!(Coeff(coeff, rhs))
            //}
            // v1 * v2 * ... * w
            (FA::SimplProd(mut vp), v @ FA::Var(_)) => {
                vp.0.push_back(v);
                FA::SimplProd(vp)
            }
            // w * v1 * v2 * ...
            (v @ FA::Var(_), FA::SimplProd(mut vp)) => {
                vp.0.push_front(v);
                FA::SimplProd(vp)
            }
            // a/b * c -> (a * c) / b
            (FA::Frac(Frac(n, d)), rhs) => fa!(Frac(*d * rhs, n)),
            // a * b/c -> (a * b) / c
            (lhs, FA::Frac(Frac(n, d))) => fa!(Frac(lhs * *d, n)),
            (FA::SimplProd(mut lhs), FA::SimplProd(rhs)) => {
                lhs.0.extend(rhs.0);
                FA::SimplProd(lhs)
            }
            // r * e -> Coeff(r, e)
            (r @ FA::Rational(_), e) | (e, r @ FA::Rational(_)) if !e.is_next_rational() => {
                fa!(Coeff(r, e))
            }
            //(v @ FA::Var(_), e @ FA::Sum(_) | e @ FA::Prod(_)) /*if var.len() == 1*/ => {
            //    e!(Coeff(v, e))
            //}
            // e1 * e2 * ... * f
            (FA::SimplProd(mut lhs), rhs) => {
                lhs.0.push_back(rhs);
                FA::SimplProd(lhs)
            }
            // f * e1 * e2 * ...
            (lhs, FA::SimplProd(mut rhs)) => {
                rhs.0.push_front(lhs);
                FA::SimplProd(rhs)
            }
            (lhs, rhs) => fa!(SimplProd([lhs, rhs])),
        }
    }
}

impl<'a> ops::Div for FmtAst<'a> {
    type Output = FmtAst<'a>;

    fn div(self, rhs: Self) -> Self::Output {
        use FmtAst as FA;
        match (self, rhs) {
            (FA::Frac(Frac(n, d)), rhs) => fa!(Frac(n, *d * rhs)),
            (rhs, FA::Frac(Frac(n, d))) => rhs * fa!(Frac(d, n)),
            (lhs, rhs) => fa!(Frac(lhs, rhs)),
        }
    }
}

impl<'a> crate::utils::Pow for FmtAst<'a> {
    type Output = FmtAst<'a>;

    fn pow(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (base, exp) if exp.is_one() => base,
            (base, exp) => fa!(Pow(base, exp)),
        }
    }
}

mod unicode {
    macro_rules! symbol {
        ($name: ident, $str:literal) => {
            pub const $name: &str = $str;
        };
    }

    symbol!(MINUS, "−");
    symbol!(UNARY_MINUS, "-");
    symbol!(PLUS, "+");
    symbol!(TIMES, "×");
    symbol!(DOT, "·");
}

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct UnicodeFmt {
    buf: String,
}

impl FormatWith<UnicodeFmt> for Var<'_> {
    fn fmt_with(&self, f: &mut UnicodeFmt) -> fmt::Result {
        write!(f.buf, "{self}")
    }
}
impl FormatWith<UnicodeFmt> for Rational {
    fn fmt_with(&self, f: &mut UnicodeFmt) -> fmt::Result {
        write!(f.buf, "{self}")
    }
}
impl FormatWith<UnicodeFmt> for Sub<'_> {
    fn fmt_with(&self, f: &mut UnicodeFmt) -> fmt::Result {
        Self::fmt_paren_prec(&self.0, f)?;
        write!(f.buf, " {} ", unicode::MINUS)?;
        Self::fmt_paren_prec(&self.1, f)
    }
}
impl FormatWith<UnicodeFmt> for Coeff<'_> {
    fn fmt_with(&self, f: &mut UnicodeFmt) -> fmt::Result {
        if self.0.is_one() {
        } else if self.0.is_min_one() {
            write!(f.buf, "{}", unicode::UNARY_MINUS)?;
        } else {
            Self::fmt_paren_prec(&self.0, f)?;
        }
        Self::fmt_paren_prec(&self.1, f)
    }
}
impl FormatWith<UnicodeFmt> for Sum<'_> {
    fn fmt_with(&self, f: &mut UnicodeFmt) -> fmt::Result {
        if self.0.is_empty() {
            return Ok(());
        }

        let mut iter = self.0.iter();
        iter.next().unwrap().fmt_with(f)?;
        iter.try_for_each(|e| {
            match e {
                FmtAst::Coeff(e) if e.is_neg() => {
                    write!(f.buf, " {} ", unicode::MINUS)?;
                    e.0.fmt_with(f)
                }
                _ => {
                    write!(f.buf, " {} ", unicode::PLUS)?;
                    e.fmt_with(f)
                }
            }
            //write!(f.buf, " {} ", unicode::PLUS)?;
            //e.fmt_with(f)
        })
    }
}
impl FormatWith<UnicodeFmt> for Prod<'_> {
    fn fmt_with(&self, f: &mut UnicodeFmt) -> fmt::Result {
        if self.0.is_empty() {
            return Ok(());
        }

        let mut iter = self.0.iter();
        Self::fmt_paren_prec(iter.next().unwrap(), f)?;
        iter.try_for_each(|e| {
            write!(f.buf, " {} ", unicode::TIMES)?;
            Self::fmt_paren_prec(e, f)
        })
    }
}
impl FormatWith<UnicodeFmt> for Frac<'_> {
    fn fmt_with(&self, f: &mut UnicodeFmt) -> fmt::Result {
        Self::fmt_paren_prec(&self.0, f)?;
        write!(f.buf, "/")?;
        Self::fmt_paren_prec(&self.1, f)
    }
}
impl FormatWith<UnicodeFmt> for Pow<'_> {
    fn fmt_with(&self, f: &mut UnicodeFmt) -> fmt::Result {
        Self::fmt_paren_prec(&self.0, f)?;
        write!(f.buf, "^")?;
        Self::fmt_paren_prec(&self.1, f)
    }
}
impl FormatWith<UnicodeFmt> for SimplProd<'_> {
    fn fmt_with(&self, f: &mut UnicodeFmt) -> fmt::Result {
        if self.0.is_empty() {
            return Ok(());
        }

        let mut iter = self.0.iter();
        Self::fmt_paren_prec(iter.next().unwrap(), f)?;
        iter.try_for_each(|e| {
            write!(f.buf, "{}", unicode::DOT)?;
            Self::fmt_paren_prec(e, f)
        })
    }
}
impl FormatWith<UnicodeFmt> for FmtAst<'_> {
    fn fmt_with(&self, f: &mut UnicodeFmt) -> fmt::Result {
        match self {
            FmtAst::Sub(n) => n.fmt_with(f),
            FmtAst::Coeff(n) => n.fmt_with(f),
            FmtAst::Rational(n) => n.fmt_with(f),
            FmtAst::Var(n) => n.fmt_with(f),
            FmtAst::Frac(n) => n.fmt_with(f),
            FmtAst::Pow(n) => n.fmt_with(f),
            FmtAst::Sum(n) => n.fmt_with(f),
            FmtAst::Prod(n) => n.fmt_with(f),
            FmtAst::SimplProd(n) => n.fmt_with(f),
            FmtAst::Undef => write!(f.buf, "undef"),
        }
    }
}

impl ExprFormatter for UnicodeFmt {
    type Result = fmt::Result;

    fn fmt_paren(&mut self, e: &impl FormatWith<Self>) -> Self::Result {
        write!(self.buf, "(")?;
        e.fmt_with(self)?;
        write!(self.buf, ")")
    }
}

impl Display for UnicodeFmt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.buf)
    }
}

#[cfg(test)]
mod test_unicode_fmt {
    use super::*;
    
    use calcurs_macros::expr as e;

    #[test]
    fn basic() {
        let fmt_res = vec![
            (e!(a + b * c), "a + b·c"),
            (e!(2 * x * y), "2x·y"),
            (e!(2 * x^3), "2x^3"),
            (e!(2 * x^(a + b)), "2x^(a + b)"),
            (e!(2 * x^(2*a)), "2x^(2a)"),
        ];

        for (e, res) in fmt_res {
            assert_eq!(e.fmt_unicode().buf, res)
        }
    }
}

//pub enum SumElementSign {
//    Plus,
//    Minus,
//}
//
//pub struct Sum2(BinaryHeap<SumElementSign>);
