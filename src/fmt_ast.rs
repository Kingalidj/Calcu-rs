use crate::{expr::{Expr, PTR}, rational::Rational};
//use num::pow::Pow as Power;
use std::{
    collections::VecDeque,
    fmt::{self, Display, Write},
    ops,
};

pub type Var = PTR<str>;

/// Specifies formatting for the [crate::Expr]
///
/// The basic Operator traits are implemented for constructing the formatted AST \
/// No calculations are performed, e.g 2 * 3 will be outputted as 2 * 3. \
/// All Primitives are stored as [Ref]
#[derive(Debug, Clone)]
pub enum FmtAst {
    Rational(Rational),
    Var(Var),
    Undef,

    Sub(Sub),
    Coeff(Coeff),
    Frac(Frac),
    Pow(Pow),
    Sum(Sum),
    Prod(Prod),
    SimplProd(SimplProd),
}

impl FmtAst {
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

    /// check for 1 / x, if yes, return x
    ///
    fn is_one_div(&self) -> Option<&FmtAst> {
        if let FmtAst::Pow(pow) = self {
            if pow.1.is_min_one() {
                return Some(&pow.0);
            }
        }

        None
    }

    fn get_one_div(&self) -> Option<FmtAst> {
        if let FmtAst::Pow(pow) = self {
            if pow.1.is_min_one() {
                return Some(*pow.0.clone());
            }
        }

        None
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
#[derive(Debug, Clone)]
pub struct Sub(Box<FmtAst>, Box<FmtAst>);

/// [Rational] * [FmtAst], e.g 3(a + b), -b^2
///
#[derive(Debug, Clone)]
pub struct Coeff(Box<FmtAst>, Box<FmtAst>);

impl Coeff {
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
#[derive(Debug, Clone)]
pub struct Frac(Box<FmtAst>, Box<FmtAst>);

/// [FmtAst] ^ [FmtAst]
///
#[derive(Debug, Clone)]
pub struct Pow(Box<FmtAst>, Box<FmtAst>);

#[derive(Debug, Default, Clone)]
pub struct SimplProd(VecDeque<FmtAst>);

/// Sum([FmtAst]), e.g: 1 + a + 2b
///
#[derive(Debug, Default, Clone)]
pub struct Sum(VecDeque<FmtAst>);

/// Prod([FmtAst]), e.g: 1 * 2 * 3c
///
#[derive(Debug, Default, Clone)]
pub struct Prod(VecDeque<FmtAst>);

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
impl_precedence!(Sum;     1);
impl_precedence!(Sub;     1);
impl_precedence!(Prod;    2);
impl_precedence!(Coeff;   2);
impl_precedence!(SimplProd; 2);
impl_precedence!(Frac;    3);
impl_precedence!(Pow;     4);
impl_precedence!(Var;     5);

impl FmtPrecedence for Rational {
    fn prec_of() -> u32 {
        5
    }
    fn prec_of_val(&self) -> u32 {
        if !self.is_int() {
            Frac::prec_of()
        } else {
            5
        }
    }
}

impl FmtPrecedence for FmtAst {
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
            FA::Undef => 5,
        }
    }
}

impl ops::Add for FmtAst {
    type Output = FmtAst;

    fn add(self, rhs: Self) -> Self::Output {
        use FmtAst as FA;
        match (self, rhs) {
            (lhs, FA::Rational(Rational::ZERO)) => lhs,

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

impl ops::Sub for FmtAst {
    type Output = FmtAst;

    fn sub(self, rhs: Self) -> Self::Output {
        fa!(Sub(self, rhs))
    }
}

impl ops::Mul for FmtAst {
    type Output = FmtAst;

    fn mul(self, rhs: Self) -> Self::Output {
        use FmtAst as FA;
        match (self, rhs) {
            (lhs, rhs) if rhs.is_one_div().is_some() => {
                lhs / rhs.is_one_div().unwrap().clone()
            }

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
            (FA::Frac(Frac(a, b)), c) => fa!(Frac(*a * c, b)),
            // a * b/c -> (a * b) / c
            (a, FA::Frac(Frac(b, c))) => fa!(Frac(a * *b, c)),
            (FA::SimplProd(mut lhs), FA::SimplProd(rhs)) => {
                lhs.0.extend(rhs.0);
                FA::SimplProd(lhs)
            }
            // r * e -> Coeff(r, e)
            (r @ FA::Rational(_), e) | (e, r @ FA::Rational(_)) if !e.is_next_rational() => {
                fa!(Coeff(r, e))
            }
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

impl ops::Div for FmtAst {
    type Output = FmtAst;

    fn div(self, rhs: Self) -> Self::Output {
        use FmtAst as FA;
        match (self, rhs) {
            (FA::Frac(Frac(n, d)), rhs) => fa!(Frac(n, *d * rhs)),
            (rhs, FA::Frac(Frac(n, d))) => rhs * fa!(Frac(d, n)),
            (lhs, rhs) => fa!(Frac(lhs, rhs)),
        }
    }
}

impl FmtAst {
    pub fn pow(self, rhs: Self) -> FmtAst {
        match (self, rhs) {
            (base, exp) if exp.is_one() => base,
            (base, exp) if exp.is_min_one() => fa!(Frac(FmtAst::Rational(Rational::ONE), base)),
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

impl FormatWith<UnicodeFmt> for Var {
    fn fmt_with(&self, f: &mut UnicodeFmt) -> fmt::Result {
        write!(f.buf, "{self}")
    }
}
impl FormatWith<UnicodeFmt> for Rational {
    fn fmt_with(&self, f: &mut UnicodeFmt) -> fmt::Result {
        write!(f.buf, "{self}")
    }
}
impl FormatWith<UnicodeFmt> for Sub {
    fn fmt_with(&self, f: &mut UnicodeFmt) -> fmt::Result {
        Self::fmt_paren_prec(&self.0, f)?;
        write!(f.buf, " {} ", unicode::MINUS)?;
        Self::fmt_paren_prec(&self.1, f)
    }
}
impl FormatWith<UnicodeFmt> for Coeff {
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
impl FormatWith<UnicodeFmt> for Sum {
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
impl FormatWith<UnicodeFmt> for Prod {
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
impl FormatWith<UnicodeFmt> for Frac {
    fn fmt_with(&self, f: &mut UnicodeFmt) -> fmt::Result {
        Self::fmt_paren_prec(&self.0, f)?;
        write!(f.buf, "/")?;
        Self::fmt_paren_prec(&self.1, f)
    }
}
impl FormatWith<UnicodeFmt> for Pow {
    fn fmt_with(&self, f: &mut UnicodeFmt) -> fmt::Result {
        Self::fmt_paren_prec(&self.0, f)?;
        write!(f.buf, "^")?;
        Self::fmt_paren_prec(&self.1, f)
    }
}
impl FormatWith<UnicodeFmt> for SimplProd {
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
impl FormatWith<UnicodeFmt> for FmtAst {
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
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
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
            (e!(a/b), "a/b"),
            (e!((x + y)/(x * y)), "(x + y)/(x·y)"),
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
