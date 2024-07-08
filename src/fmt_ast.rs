use calcu_rs::Rational;
use std::{
    cell::Ref,
    collections::VecDeque,
    fmt::{self, Display},
    ops,
    collections::{BinaryHeap}
};
use std::cmp::Ordering;

// TODO: Sorting for Sum, Prod, VarProd

/// Specifies formatting for the [crate::Expr]
///
/// The basic Operator traits are implemented for constructing the formatted AST \
/// No calculations are performed, e.g 2 * 3 will be outputted as 2 * 3. \
/// All Primitives are stored as [Ref]
#[derive(Debug)]
pub enum FmtAst<'a> {
    Sub(Sub<'a>),
    Coeff(Coeff<'a>),
    Atom(Atom<'a>),
    Frac(Frac<'a>),
    Pow(Pow<'a>),
    Sum(Sum<'a>),
    Prod(Prod<'a>),
    VarProd(VarProd<'a>),
}

pub type Var<'a> = &'a str;

/// Holds references to primitive [crate::Node]
///
#[derive(Debug)]
pub enum Atom<'a> {
    Rational(Ref<'a, Rational>),
    Var(Var<'a>),
    Undefined,
}

/// - [FmtAst]
///
#[derive(Debug)]
pub struct Sub<'a>(Box<FmtAst<'a>>, Box<FmtAst<'a>>);

/// [Rational] * [FmtAst], e.g 3(a + b), -b^2
///
#[derive(Debug)]
pub struct Coeff<'a>(Ref<'a, Rational>, Box<FmtAst<'a>>);

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
#[derive(Debug)]
pub struct VarProd<'a>(VecDeque<Var<'a>>);

/// Sum([FmtAst]), e.g: 1 + a + 2b
///
#[derive(Debug)]
pub struct Sum<'a>(VecDeque<FmtAst<'a>>);

/// Prod([FmtAst]), e.g: 1 * 2 * 3c
///
#[derive(Debug)]
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

macro_rules! e {
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

impl<'a> ExprFormatter for fmt::Formatter<'a> {
    type Result = fmt::Result;

    fn fmt_paren(&mut self, e: &impl FormatWith<Self>) -> Self::Result {
        write!(self, "(")?;
        e.fmt_with(self)?;
        write!(self, ")")
    }
}

impl FormatWith<fmt::Formatter<'_>> for Atom<'_> {
    fn fmt_with(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Atom::Rational(r) => write!(f, "{r}"),
            Atom::Var(v) => write!(f, "{}", v),
            Atom::Undefined => write!(f, "undef"),
        }
    }
}
impl FormatWith<fmt::Formatter<'_>> for Sub<'_> {
    fn fmt_with(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        Self::fmt_paren_prec(&self.0, f)?;
        write!(f, " - ")?;
        Self::fmt_paren_prec(&self.1, f)
    }
}
impl FormatWith<fmt::Formatter<'_>> for Coeff<'_> {
    fn fmt_with(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if &*self.0 == &Rational::ONE {
        } else if &*self.0 == &Rational::MINUS_ONE {
            write!(f, "-")?;
        } else {
            write!(f, "{}", self.0)?;
        }
        Self::fmt_paren_prec(&self.1, f)
    }
}
impl FormatWith<fmt::Formatter<'_>> for Sum<'_> {
    fn fmt_with(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.0.is_empty() {
            return Ok(());
        }

        let mut iter = self.0.iter();
        iter.next().unwrap().fmt_with(f)?;
        iter.try_for_each(|e| {
            if let FmtAst::Sub(e) = e {
                write!(f, " - ")?;
                e.0.fmt_with(f)
            } else {
                write!(f, " + ")?;
                e.fmt_with(f)
            }
        })
    }
}
impl FormatWith<fmt::Formatter<'_>> for Prod<'_> {
    fn fmt_with(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.0.is_empty() {
            return Ok(());
        }

        let mut iter = self.0.iter();
        Self::fmt_paren_prec(iter.next().unwrap(), f)?;
        iter.try_for_each(|e| {
            write!(f, " * ")?;
            Self::fmt_paren_prec(e, f)
        })
    }
}
impl FormatWith<fmt::Formatter<'_>> for Frac<'_> {
    fn fmt_with(&self, f: &mut fmt::Formatter) -> fmt::Result {
        Self::fmt_paren_prec(&self.0, f)?;
        write!(f, "/")?;
        Self::fmt_paren_prec(&self.1, f)
    }
}
impl FormatWith<fmt::Formatter<'_>> for Pow<'_> {
    fn fmt_with(&self, f: &mut fmt::Formatter) -> fmt::Result {
        Self::fmt_paren_prec(&self.0, f)?;
        write!(f, "^")?;
        Self::fmt_paren_prec(&self.1, f)
    }
}
impl FormatWith<fmt::Formatter<'_>> for VarProd<'_> {
    fn fmt_with(&self, f: &mut fmt::Formatter) -> fmt::Result {
        //if let Some(c) = &self.0 {
        //    write!(f, "{}", c)?;
        //}
        self.0.iter().try_for_each(|v| write!(f, "{v}"))
    }
}
impl FormatWith<fmt::Formatter<'_>> for FmtAst<'_> {
    fn fmt_with(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            FmtAst::Sub(n) => n.fmt_with(f),
            FmtAst::Coeff(n) => n.fmt_with(f),
            FmtAst::Atom(n) => n.fmt_with(f),
            FmtAst::Frac(n) => n.fmt_with(f),
            FmtAst::Pow(n) => n.fmt_with(f),
            FmtAst::Sum(n) => n.fmt_with(f),
            FmtAst::Prod(n) => n.fmt_with(f),
            FmtAst::VarProd(n) => n.fmt_with(f),
        }
    }
}

impl Display for FmtAst<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.fmt_with(f)
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
impl_precedence!(VarProd<'_>; 2);
impl_precedence!(Pow<'_>;     3);

impl FmtPrecedence for Atom<'_> {
    fn prec_of() -> u32 {
        4
    }
    fn prec_of_val(&self) -> u32 {
        match self {
            Atom::Rational(r) if !r.is_int() => Frac::prec_of(),
            _ => 4,
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
            FA::Atom(x) => x.prec_of_val(),
            FA::Frac(x) => x.prec_of_val(),
            FA::Pow(x) => x.prec_of_val(),
            FA::Sum(x) => x.prec_of_val(),
            FA::Prod(x) => x.prec_of_val(),
            FA::VarProd(x) => x.prec_of_val(),
        }
    }
}

impl<'a> ops::Add for FmtAst<'a> {
    type Output = FmtAst<'a>;

    fn add(self, rhs: Self) -> Self::Output {
        use FmtAst as FA;
        match (self, rhs) {
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
            (lhs, rhs) => e!(Sum([lhs, rhs])),
        }
    }
}

impl<'a> ops::Sub for FmtAst<'a> {
    type Output = FmtAst<'a>;

    fn sub(self, rhs: Self) -> Self::Output {
        e!(Sub(self, rhs))
    }
}

impl<'a> ops::Mul for FmtAst<'a> {
    type Output = FmtAst<'a>;

    fn mul(self, rhs: Self) -> Self::Output {
        use FmtAst as FA;
        match (self, rhs) {
            // var1 * var2
            (FA::Atom(Atom::Var(v1)), FA::Atom(Atom::Var(v2))) => e!(VarProd([v1, v2])),
            // coeff(c, e1) * e2 -> coeff(c, e1 * e2)
            (FA::Coeff(Coeff(coeff, expr)), rhs) => {
                let mut lhs = expr;
                *lhs = *lhs * rhs;
                e!(Coeff(coeff, lhs))
            }
            // e1 * coeff(c, e2) -> coeff(c, e1 * e2)
            (lhs, FA::Coeff(Coeff(coeff, expr))) => {
                let mut rhs = expr;
                *rhs = lhs * *rhs;
                e!(Coeff(coeff, rhs))
            }
            // v1 * v2 * ... * w
            (FA::VarProd(mut vp), FA::Atom(Atom::Var(v))) => {
                vp.0.push_back(v);
                FA::VarProd(vp)
            }
            // w * v1 * v2 * ...
            (FA::Atom(Atom::Var(v)), FA::VarProd(mut vp)) => {
                vp.0.push_front(v);
                FA::VarProd(vp)
            }
            // a/b * c -> (a * c) / b
            (FA::Frac(Frac(n, d)), rhs) => e!(Frac(*d * rhs, n)),
            // a * b/c -> (a * b) / c
            (lhs, FA::Frac(Frac(n, d))) => e!(Frac(lhs * *d, n)),
            (FA::Prod(mut lhs), FA::Prod(rhs)) => {
                lhs.0.extend(rhs.0);
                FA::Prod(lhs)
            }
            // r * e -> Coeff(r, e)
            (FA::Atom(Atom::Rational(r)), e)
            | (e, FA::Atom(Atom::Rational(r))) => {
                e!(Coeff(r, e))
            }
            // e1 * e2 * ... * f
            (FA::Prod(mut lhs), rhs) => {
                lhs.0.push_back(rhs);
                FA::Prod(lhs)
            }
            // f * e1 * e2 * ...
            (lhs, FA::Prod(mut rhs)) => {
                rhs.0.push_front(lhs);
                FA::Prod(rhs)
            }
            (lhs, rhs) => e!(Prod([lhs, rhs])),
        }
    }
}

impl<'a> ops::Div for FmtAst<'a> {
    type Output = FmtAst<'a>;

    fn div(self, rhs: Self) -> Self::Output {
        use FmtAst as FA;
        match (self, rhs) {
            (FA::Frac(Frac(n, d)), rhs) => e!(Frac(n, *d * rhs)),
            (rhs, FA::Frac(Frac(n, d))) => rhs * e!(Frac(d, n)),
            (lhs, rhs) => e!(Frac(lhs, rhs)),
        }
    }
}

impl<'a> crate::Pow for FmtAst<'a> {
    type Output = FmtAst<'a>;

    fn pow(self, rhs: Self) -> Self::Output {
        e!(Pow(self, rhs))
    }
}

pub enum SumElementSign {
    Plus,
    Minus,
}


pub struct Sum2(BinaryHeap<SumElementSign>);

