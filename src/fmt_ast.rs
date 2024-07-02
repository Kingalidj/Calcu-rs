use calcu_rs::Rational;
use std::{
    cell::Ref,
    collections::VecDeque,
    fmt::{self, Display, Formatter},
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

/// Allows for custom formatting by implementing a format function for every [FmtAst] variant
///
pub trait ExprFormatter {
    type Result;
    fn fmt_atom(&mut self, atom: &Atom) -> Self::Result;
    /// helper function, since parenthesis placement is decided by the [ExprFormatter], not [FmtAst]
    fn fmt_paren(&mut self, e: &FmtAst, cond: impl Fn(&FmtAst) -> bool) -> Self::Result;
    fn fmt_sum(&mut self, s: &Sum) -> Self::Result;
    fn fmt_sub(&mut self, s: &Sub) -> Self::Result;
    fn fmt_prod(&mut self, p: &Prod) -> Self::Result;
    fn fmt_frac(&mut self, frac: &Frac) -> Self::Result;
    fn fmt_pow(&mut self, pow: &Pow) -> Self::Result;
    fn fmt_var_prod(&mut self, v: &VarProd) -> Self::Result;
    /// calls self.fmt_xxx(fa) on each variant by default
    fn fmt_expr(&mut self, fa: &FmtAst) -> Self::Result {
        match fa {
            FmtAst::Sub(e) => self.fmt_sub(e),
            FmtAst::Atom(a) => self.fmt_atom(a),
            FmtAst::Frac(f) => self.fmt_frac(f),
            FmtAst::Pow(p) => self.fmt_pow(p),
            FmtAst::Sum(s) => self.fmt_sum(s),
            FmtAst::Prod(p) => self.fmt_prod(p),
            FmtAst::VarProd(vp) => self.fmt_var_prod(vp),
        }
    }
}

/// an [ExprFormatter] that is used in [Display::fmt]
///
pub struct ExprDisplay<'a, 'b> {
    f: &'a mut Formatter<'b>,
}

impl<'a, 'b> ExprFormatter for ExprDisplay<'a, 'b> {
    type Result = fmt::Result;

    fn fmt_atom(&mut self, atom: &Atom) -> Self::Result {
        match atom {
            Atom::Rational(r) => write!(self.f, "{r}"),
            Atom::Var(v) => write!(self.f, "{}", v),
            Atom::Undefined => write!(self.f, "undef"),
        }
    }

    fn fmt_paren(&mut self, e: &FmtAst, cond: impl Fn(&FmtAst) -> bool) -> Self::Result {
        let use_paren = cond(e);
        if use_paren {
            write!(self.f, "(")?;
        }
        self.fmt_expr(e)?;
        if use_paren {
            write!(self.f, ")")?;
        }
        Ok(())
    }

    fn fmt_sum(&mut self, rs: &Sum) -> Self::Result {
        if rs.0.is_empty() {
            return Ok(());
        }

        let mut iter = rs.0.iter();
        self.fmt_expr(iter.next().unwrap())?;
        iter.try_for_each(|e| {
            write!(self.f, " + ")?;
            self.fmt_expr(e)
        })
    }

    fn fmt_sub(&mut self, sub: &Sub) -> Self::Result {
        write!(self.f, "-")?;
        self.fmt_expr(&sub.0)
    }

    fn fmt_prod(&mut self, rs: &Prod) -> Self::Result {
        if rs.0.is_empty() {
            return Ok(());
        }

        let mut iter = rs.0.iter();
        self.fmt_expr(iter.next().unwrap())?;
        iter.try_for_each(|e| {
            write!(self.f, " * ")?;
            self.fmt_expr(e)
        })
    }

    fn fmt_frac(&mut self, frac: &Frac) -> Self::Result {
        self.fmt_paren(&frac.0, |e| e.prec_of_val() < Frac::prec_of())?;
        write!(self.f, "/")?;
        self.fmt_paren(&frac.1, |e| e.prec_of_val() < Frac::prec_of())
    }

    fn fmt_pow(&mut self, pow: &Pow) -> Self::Result {
        self.fmt_paren(&pow.0, |e| e.prec_of_val() < Pow::prec_of())?;
        write!(self.f, "^")?;
        self.fmt_paren(&pow.1, |e| e.prec_of_val() < Pow::prec_of())
    }

    fn fmt_var_prod(&mut self, vp: &VarProd) -> Self::Result {
        if let Some(c) = &vp.0 {
            write!(self.f, "{}", c)?;
        }
        vp.1.iter().try_for_each(|v| write!(self.f, "{v}"))
    }
}

impl Display for FmtAst<'_> {
    fn fmt(&self, fmt: &mut Formatter<'_>) -> fmt::Result {
        ExprDisplay { f: fmt }.fmt_expr(self)
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
impl_precedence!(Sum<'_>; 1);
impl_precedence!(Prod<'_>; 2);
impl_precedence!(Frac<'_>; 2);
impl_precedence!(Pow<'_>; 3);
// ...
impl_precedence!(Sub<'_>; u32::MAX - 2);
impl_precedence!(VarProd<'_>; u32::MAX - 1);
impl_precedence!(Atom<'_>; u32::MAX);

impl FmtPrecedence for FmtAst<'_> {
    fn prec_of() -> u32 {
        0
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
            | (E::VarProd(mut vp), E::Atom(Atom::Rational(r))) if vp.0.is_none() => {
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
