use calcu_rs::Rational;
use paste::paste;
use std::{
    cell::Ref,
    collections::VecDeque,
    fmt::{self, Display},
    ops,
};

// TODO: Sorting for Sum, Prod, VarProd

/// Specifies formatting for the [crate::Expr]
///
/// The basic Operator traits are implemented for constructing the formatted AST \
/// No calculations are performed, e.g 2 * 3 will be outputted as 2 * 3. \
/// All Primitives are stored as [Ref]
#[derive(Debug)]
pub enum FmtAst<'a> {
    Sub(Sub<'a>),
    Atom(Atom<'a>),
    Frac(Frac<'a>),
    Pow(Pow<'a>),
    Sum(Sum<'a>),
    Prod(Prod<'a>),
    VarProd(VarProd<'a>),
}

macro_rules! e {
    ($ty:ident($e:expr)) => {
        FmtAst::$ty($ty($e.into()).into())
    };
    ($ty:ident($e1:expr, $e2:expr)) => {
        FmtAst::$ty($ty($e1.into(), $e2.into()))
    };
}

/// - [FmtAst]
///
#[derive(Debug)]
pub struct Sub<'a>(Box<FmtAst<'a>>);

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
pub struct VarProd<'a>(Option<Ref<'a, Rational>>, VecDeque<Var<'a>>);

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
        write!(f, "-")?;
        Self::fmt_paren_prec(&self.0, f)
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
        if let Some(c) = &self.0 {
            write!(f, "{}", c)?;
        }
        self.1.iter().try_for_each(|v| write!(f, "{v}"))
    }
}
impl FormatWith<fmt::Formatter<'_>> for FmtAst<'_> {
    fn fmt_with(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            FmtAst::Sub(n) => n.fmt_with(f),
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

pub type Var<'a> = &'a str;

/// Holds references to primitive [crate::Node]
///
#[derive(Debug)]
pub enum Atom<'a> {
    Rational(Ref<'a, Rational>),
    Var(Var<'a>),
    Undefined,
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
impl_precedence!(Prod<'_>;    2);
impl_precedence!(Sub<'_>;     2);
impl_precedence!(Frac<'_>;    2);
impl_precedence!(VarProd<'_>; 2);
impl_precedence!(Pow<'_>;     3);
//impl_precedence!(Atom<'_>;    4);

impl FmtPrecedence for Atom<'_> {
    fn prec_of() -> u32 {
        4
    }
    fn prec_of_val(&self) -> u32 {
        match self {
            Atom::Rational(r) if !r.is_int() => Frac::prec_of(),
            _ => 4
        }
    }
}

impl FmtPrecedence for FmtAst<'_> {
    fn prec_of() -> u32 {
        panic!("FmtAst precedence is defined when created")
    }
    fn prec_of_val(&self) -> u32 {
        use FmtAst as E;
        match self {
            E::Sub(x) => x.prec_of_val(),
            E::Atom(x) => x.prec_of_val(),
            E::Frac(x) => x.prec_of_val(),
            E::Pow(x) => x.prec_of_val(),
            E::Sum(x) => x.prec_of_val(),
            E::Prod(x) => x.prec_of_val(),
            E::VarProd(x) => x.prec_of_val(),
        }
    }
}

impl<'a> ops::Add for FmtAst<'a> {
    type Output = FmtAst<'a>;

    fn add(self, rhs: Self) -> Self::Output {
        use FmtAst as E;
        match (self, rhs) {
            (E::Sum(mut lhs), E::Sum(rhs)) => {
                lhs.0.extend(rhs.0);
                E::Sum(lhs)
            }
            (lhs, E::Sum(mut rhs)) => {
                rhs.0.push_front(lhs);
                E::Sum(rhs)
            }
            (E::Sum(mut lhs), rhs) => {
                lhs.0.push_back(rhs);
                E::Sum(lhs)
            }
            (lhs, rhs) => e!(Sum([lhs, rhs])),
        }
    }
}

impl<'a> ops::Sub for FmtAst<'a> {
    type Output = FmtAst<'a>;

    fn sub(self, rhs: Self) -> Self::Output {
        use FmtAst as E;
        match (self, rhs) {
            (E::Sum(mut lhs), rhs) => {
                lhs.0.push_back(e!(Sub(rhs)));
                E::Sum(lhs)
            }
            (lhs, E::Sum(mut rhs)) => {
                rhs.0.push_front(e!(Sub(lhs)));
                E::Sum(rhs)
            }
            (lhs, rhs) => e!(Sum([lhs, e!(Sub(rhs))])),
        }
    }
}

impl<'a> ops::Mul for FmtAst<'a> {
    type Output = FmtAst<'a>;

    fn mul(self, rhs: Self) -> Self::Output {
        use FmtAst as E;
        match (self, rhs) {
            // v1 * v2
            (E::Atom(Atom::Var(v1)), E::Atom(Atom::Var(v2))) => {
                e!(VarProd(None, [v1, v2]))
            }
            (e, E::Atom(Atom::Rational(r))) | (E::Atom(Atom::Rational(r)), e)
                if &*r == &Rational::MINUS_ONE =>
            {
                e!(Sub(e))
            }

            (E::Sub(sub), rhs) => {
                let mut lhs = sub.0;
                *lhs = *lhs * rhs;
                e!(Sub(lhs))
            }
            (lhs, E::Sub(sub)) => {
                let mut rhs = sub.0;
                *rhs = lhs * *rhs;
                e!(Sub(rhs))
            }
            // r * v or v * r
            (E::Atom(Atom::Rational(r)), E::Atom(Atom::Var(v)))
            | (E::Atom(Atom::Var(v)), E::Atom(Atom::Rational(r))) => {
                e!(VarProd(r, [v]))
            }
            (E::VarProd(mut vp), E::Atom(Atom::Var(v))) => {
                vp.1.push_back(v);
                E::VarProd(vp)
            }
            (E::Atom(Atom::Var(v)), E::VarProd(mut vp)) => {
                vp.1.push_front(v);
                E::VarProd(vp)
            }
            (E::Atom(Atom::Rational(r)), E::VarProd(mut vp))
            | (E::VarProd(mut vp), E::Atom(Atom::Rational(r)))
                if vp.0.is_none() =>
            {
                vp.0 = Some(r);
                E::VarProd(vp)
            }
            (E::Frac(Frac(n, d)), rhs) => {
                e!(Frac(*d * rhs, n))
            }
            (lhs, E::Frac(Frac(n, d))) => {
                e!(Frac(lhs * *d, n))
            }
            (E::Prod(mut lhs), E::Prod(rhs)) => {
                lhs.0.extend(rhs.0);
                E::Prod(lhs)
            }
            (E::Prod(mut lhs), rhs) => {
                lhs.0.push_back(rhs);
                E::Prod(lhs)
            }
            (lhs, E::Prod(mut rhs)) => {
                rhs.0.push_front(lhs);
                E::Prod(rhs)
            }
            (lhs, rhs) => e!(Prod([lhs, rhs])),
        }
    }
}

impl<'a> ops::Div for FmtAst<'a> {
    type Output = FmtAst<'a>;

    fn div(self, rhs: Self) -> Self::Output {
        use FmtAst as E;
        match (self, rhs) {
            (E::Frac(Frac(n, d)), rhs) => {
                e!(Frac(n, *d * rhs))
            }
            (rhs, E::Frac(Frac(n, d))) => rhs * e!(Frac(d, n)),
            (lhs, rhs) => {
                e!(Frac(lhs, rhs))
            }
        }
    }
}

impl<'a> crate::Pow for FmtAst<'a> {
    type Output = FmtAst<'a>;

    fn pow(self, rhs: Self) -> Self::Output {
        e!(Pow(self, rhs))
    }
}
