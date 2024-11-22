use crate::{
    polynomial::{MonomialView, PolynomialView, VarSet},
    rational::Rational,
    sym_fmt,
    utils::{log_macros::*, HashSet},
};
use std::{borrow::Borrow, cmp, fmt, hash, ops, slice};

use derive_more::{Debug, Display, From, Into, IsVariant, TryUnwrap, Unwrap};
use paste::paste;
use serde::{Deserialize, Serialize};

//pub(crate) type PTR<T> = std::sync::Arc<T>;
pub(crate) type PTR<T> = std::rc::Rc<T>;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct Derivative {
    arg: Expr,
    var: Expr,
    degree: u64,
}

#[derive(
    Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Display, Debug, From, Serialize, Deserialize,
)]
#[from(&str, String)]
#[debug("{_0}")]
pub struct Var(pub(crate) PTR<str>);

#[derive(
    Clone, PartialEq, Eq, Hash, Debug, From, IsVariant, Unwrap, TryUnwrap, Serialize, Deserialize,
)]
#[unwrap(ref)]
#[try_unwrap(ref)]
pub enum Atom {
    #[debug("undef")]
    Undef,
    #[from(i32, i64, u32, u64, i128, Rational)]
    #[debug("{_0:?}")]
    Rational(Rational),
    #[from]
    #[debug("{_0:?}")]
    Irrational(Irrational),
    #[from(forward)]
    #[debug("{_0:?}")]
    Var(Var),
    #[from]
    #[debug("{_0:?}")]
    Sum(Sum),
    #[from]
    #[debug("{_0:?}")]
    Prod(Prod),
    #[from]
    #[debug("{_0:?}")]
    Pow(Pow),
    #[from]
    #[debug("{_0:?}")]
    Func(Func),
}

impl fmt::Display for Atom {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.fmt_ast())
    }
}

impl Atom {
    pub const UNDEF: Atom = Atom::Undef;

    pub const MINUS_TWO: Atom = Atom::Rational(Rational::MINUS_TWO);
    pub const MINUS_ONE: Atom = Atom::Rational(Rational::MINUS_ONE);
    pub const ZERO: Atom = Atom::Rational(Rational::ZERO);
    pub const ONE: Atom = Atom::Rational(Rational::ONE);
    pub const TWO: Atom = Atom::Rational(Rational::TWO);

    pub const PI: Atom = Atom::Irrational(Irrational::PI);
    pub const E: Atom = Atom::Irrational(Irrational::E);

    pub fn is_zero(&self) -> bool {
        self == &Atom::ZERO
    }
    pub fn is_one(&self) -> bool {
        self == &Atom::ONE
    }
    pub fn is_min_one(&self) -> bool {
        self == &Atom::MINUS_ONE
    }
    pub fn is_pi(&self) -> bool {
        self == &Atom::PI
    }
    pub fn is_e(&self) -> bool {
        self == &Atom::E
    }
    pub fn is_neg(&self) -> bool {
        self.is_rational_and(Rational::is_neg)
    }
    pub fn is_pos(&self) -> bool {
        self.is_rational_and(Rational::is_pos)
    }
    pub fn is_int(&self) -> bool {
        self.is_rational_and(Rational::is_int)
    }
    pub fn is_even(&self) -> bool {
        self.is_rational_and(Rational::is_even)
    }
    pub fn is_odd(&self) -> bool {
        self.is_rational_and(|r| r.is_int() && !r.is_even())
    }
    pub fn is_number(&self) -> bool {
        self.is_real()
    }
    pub fn is_real(&self) -> bool {
        self.is_rational() || self.is_irrational()
    }
    pub fn is_irreducible(&self) -> bool {
        match self {
            Atom::Undef | Atom::Rational(_) | Atom::Irrational(_) | Atom::Var(_) => true,

            Atom::Func(_) | Atom::Sum(_) | Atom::Prod(_) | Atom::Pow(_) => false,
        }
    }
    pub fn is_rational_and(&self, cond: impl Fn(&Rational) -> bool) -> bool {
        match self {
            Atom::Rational(r) => cond(r),
            _ => false,
        }
    }
    pub fn is_const(&self) -> bool {
        self.is_number()
    }
    pub fn is_sin(&self) -> bool {
        match self {
            Atom::Func(f) => f.is_sin(),
            _ => false,
        }
    }
    pub fn is_cos(&self) -> bool {
        match self {
            Atom::Func(f) => f.is_cos(),
            _ => false,
        }
    }
    //pub(crate) fn is_coeff(&self) -> bool {
    //    self.is_rational() || self.is_undef()
    //}
    //pub(crate) fn is_term(&self) -> bool {
    //    !self.is_coeff()
    //}

    pub fn try_unwrap_int(&self) -> Option<i128> {
        match self {
            Atom::Rational(r) if r.is_int() => Some(r.numer().into()),
            _ => None,
        }
    }
    pub fn unwrap_int(&self) -> i128 {
        self.try_unwrap_int().unwrap()
    }

    pub fn try_as_real(&self) -> Option<Real> {
        match self {
            Atom::Rational(r) => Some(Real::Rational(*r)),
            Atom::Irrational(i) => Some(Real::Irrational(*i)),
            _ => None,
        }
    }

    pub fn for_each_arg<'a>(&'a self, func: impl FnMut(&'a Expr)) {
        self.args().iter().for_each(func)
    }

    pub fn fmt_ast(&self) -> sym_fmt::FmtAtom {
        sym_fmt::FmtAtom::from(self)
    }
}

impl PartialOrd for Atom {
    fn partial_cmp(&self, other: &Self) -> Option<cmp::Ordering> {
        Some(self.cmp(other))
    }
}

fn cmp_slice_rev(lhs: &[Expr], rhs: &[Expr]) -> cmp::Ordering {
    let args = lhs.iter().rev().zip(rhs.iter().rev());

    for (l, r) in args {
        if !l.cmp(r).is_eq() {
            return l.cmp(r)
        }
    }

    lhs.len().cmp(&rhs.len())
}
fn cmp_slice(lhs: &[Expr], rhs: &[Expr]) -> cmp::Ordering {
    let args = lhs.iter().zip(rhs.iter());

    for (l, r) in args {
        if !l.cmp(r).is_eq() {
            return l.cmp(r)
        }
    }

    lhs.len().cmp(&rhs.len())
}

impl Ord for Atom {
    fn cmp(&self, other: &Self) -> cmp::Ordering {
        use Atom as A;
        let (lhs, rhs) = (self, other);

        let (r1, r2) = (lhs.try_as_real(), rhs.try_as_real());
        if let (Some(l), Some(r)) = (&r1, &r2) {
            return l.cmp(r);
        } else if r1.is_some() {
            return cmp::Ordering::Less
        } else if r2.is_some() {
            return cmp::Ordering::Greater
        }

        match (lhs, rhs) {
            (A::Var(l), A::Var(r)) => l.cmp(r),
            (A::Prod(_), A::Prod(_)) | (A::Sum(_), A::Sum(_)) => {
                cmp_slice_rev(lhs.args(), rhs.args())
            }
            (A::Pow(p1), A::Pow(p2)) => {
                if p1.base() != p2.base() {
                    p1.base().cmp(p2.base())
                } else {
                    p1.exponent().cmp(p2.exponent())
                }
            }
            (A::Func(f1), A::Func(f2)) => {
                let (n1, n2) = (f1.name(), f2.name());
                if n1 != n2 {
                    n1.cmp(&n2)
                } else {
                    cmp_slice(f1.args(), f2.args())
                }
            }
            (A::Prod(_) | A::Sum(_), _) => {
                if lhs.n_args() == 0 {
                    cmp::Ordering::Less
                } else if lhs.args().last().unwrap().atom() != rhs {
                    lhs.args().last().unwrap().atom().cmp(rhs)
                } else {
                    cmp::Ordering::Greater
                }
            }
            (A::Pow(p), _) => {
                if p.base().atom() != rhs {
                    p.base().atom().cmp(rhs)
                } else {
                    p.exponent().atom().cmp(&A::ONE)
                }
            }
            (A::Func(f), A::Var(v)) => {
                let n = f.name();
                if n == *v.0 {
                    cmp::Ordering::Greater
                } else {
                    n.as_str().cmp(&v.0)
                }
            }
            (_, _) => {
                rhs.cmp(lhs).reverse()
            }
        }
    }
}

#[derive(Clone, PartialEq, Eq, Hash, Debug, Display, From, Serialize, Deserialize)]
#[from(forward)]
pub enum Real {
    #[debug("{_0}")]
    Rational(Rational),
    #[debug("{_0}")]
    Irrational(Irrational),
}

impl PartialOrd for Real {
    fn partial_cmp(&self, other: &Self) -> Option<cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Real {
    fn cmp(&self, other: &Self) -> cmp::Ordering {
        use ordered_float::OrderedFloat;
        OrderedFloat(self.f64_approx()).cmp(&OrderedFloat(other.f64_approx()))
    }
}

impl Real {
    pub const E: Real = Real::Irrational(Irrational::E);
    pub const PI: Real = Real::Irrational(Irrational::PI);

    pub fn f64_approx(&self) -> f64 {
        match self {
            Real::Rational(r) => r.f64_approx(),
            Real::Irrational(i) => i.f64_approx(),
        }
    }
}

impl PartialEq<Atom> for Real {
    fn eq(&self, other: &Atom) -> bool {
        match self {
            Real::Rational(r) => other.try_unwrap_rational_ref() == Ok(r),
            Real::Irrational(i) => other.try_unwrap_irrational_ref() == Ok(i),
        }
    }
}
impl PartialEq<Real> for Atom {
    fn eq(&self, other: &Real) -> bool {
        other == self
    }
}

impl From<Real> for Atom {
    fn from(value: Real) -> Self {
        match value {
            Real::Rational(r) => r.into(),
            Real::Irrational(i) => i.into(),
        }
    }
}

#[derive(
    Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Display, Serialize, Deserialize,
)]
pub enum Irrational {
    #[debug("e")]
    #[display("𝓮")]
    E,
    #[debug("pi")]
    #[display("{}", unicode::pi())]
    PI,
}

impl Irrational {
    pub fn f64_approx(&self) -> f64 {
        match self {
            Irrational::E => std::f64::consts::E,
            Irrational::PI => std::f64::consts::PI,
        }
    }
}

#[derive(
    Clone,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    Debug,
    Display,
    Unwrap,
    TryUnwrap,
    Serialize,
    Deserialize,
    IsVariant,
)]
#[unwrap(ref)]
#[try_unwrap(ref)]
#[display("{}({_0})", self.name())]
pub enum Func {
    Sin(Expr),
    ArcSin(Expr),

    Cos(Expr),
    ArcCos(Expr),

    Tan(Expr),
    ArcTan(Expr),

    Sec(Expr),
    ArcSec(Expr),

    Cot(Expr),
    ArcCot(Expr),

    // 1 / sin(x)
    Csc(Expr),
    ArcCsc(Expr),

    //Exp(Expr),
    #[display("{}{_1}", self.name())]
    Log(Real, Expr),
}

impl Func {
    pub fn is_nat_log(&self) -> bool {
        self.try_unwrap_nat_log_ref().is_some()
    }

    pub fn is_trig(&self) -> bool {
        match self {
            Func::Sin(_)
            | Func::ArcSin(_)
            | Func::Cos(_)
            | Func::ArcCos(_)
            | Func::Tan(_)
            | Func::ArcTan(_)
            | Func::Sec(_)
            | Func::ArcSec(_)
            | Func::Cot(_)
            | Func::ArcCot(_)
            | Func::Csc(_)
            | Func::ArcCsc(_) => true,
            _ => false,
        }
    }

    pub fn try_unwrap_log_base(&self, base: &Real) -> Option<&Expr> {
        match self.try_unwrap_log_ref() {
            Ok((b, expr)) if b == base => Some(expr),
            _ => None,
        }
    }

    pub fn try_unwrap_nat_log_ref(&self) -> Option<&Expr> {
        self.try_unwrap_log_base(&Real::Irrational(Irrational::E))
    }

    pub fn name(&self) -> String {
        match self {
            Func::Sin(_) => "sin",
            Func::ArcSin(_) => "arcsin",
            Func::Cos(_) => "cos",
            Func::ArcCos(_) => "arccos",
            Func::Tan(_) => "tan",
            Func::ArcTan(_) => "arctan",
            Func::Sec(_) => "sec",
            Func::ArcSec(_) => "arcsec",
            Func::Cot(_) => "cot",
            Func::ArcCot(_) => "arccot",
            Func::Csc(_) => "csc",
            Func::ArcCsc(_) => "arccsc",
            //Func::Exp(_) => "exp",
            Func::Log(Real::Irrational(Irrational::E), _) => "ln".into(),
            Func::Log(Real::Rational(r), _) if r == &Rational::from(10) => "log".into(),
            Func::Log(base, _) => return format!("log{base}"),
        }
        .into()
    }
    //pub fn iter_args_mut(&mut self) -> impl Iterator<Item = &mut Expr> {
    //    self.args_mut().iter_mut()
    //}

    pub fn derivative(&self, x: impl Borrow<Expr>) -> Expr {
        use Expr as E;
        use Func as F;

        let r = |n: i32, d: i32| E::from(Rational::from((n, d)));
        let e = |e: i32| match e {
            -1 => E::min_one(),
            1 => E::one(),
            2 => E::two(),
            _ => E::from(e),
        };
        let d = |e: &E| -> E { e.derivative(x) };

        match self {
            F::Sin(f) => d(f) * E::cos(f),
            F::Cos(f) => e(-1) * d(f) * E::sin(f),
            F::Tan(f) => d(f) * E::pow(E::sec(f), e(2)),
            F::Sec(f) => d(f) * E::tan(f) * E::sec(f),
            F::ArcSin(f) => d(f) * E::pow(e(1) - E::pow(f, e(2)), r(-1, 2)),
            F::ArcCos(f) => d(f) * e(-1) * E::pow(e(1) - E::pow(f, e(2)), r(-1, 2)),
            F::ArcTan(f) => d(f) * E::pow(e(1) + E::pow(f, e(2)), e(-1)),
            F::ArcSec(f) => d(&E::arc_cos(e(1) / f)),
            F::Cot(f) => d(f) * e(-1) * E::pow(E::csc(f), e(2)),
            F::ArcCot(f) => e(-1) * d(f) / (E::pow(f, e(2)) + e(1)),
            F::Csc(f) => d(f) * e(-1) * E::cot(f) * E::csc(f),
            F::ArcCsc(f) => {
                e(-1) * d(f) / (E::sqrt(e(1) - e(1) / E::pow(f, e(2))) * E::pow(f, e(2)))
            }
            F::Log(base, f) => d(f) * E::pow(f * E::ln(E::from(base.clone())), e(-1)),
            //F::Exp(f) => E::exp(f) * d(f),
        }
    }
}

#[derive(Clone, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct Sum {
    pub args: Vec<Expr>,
}

impl fmt::Debug for Sum {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.n_args() == 0 {
            return write!(f, "Sum[]");
        } else if self.n_args() == 1 {
            return write!(f, "Sum[{:?}]", self.args[0]);
        }
        let mut args = self.args.iter();

        if let Some(a) = args.next() {
            write!(f, "Sum[{a:?}")?;
        }

        for a in args {
            write!(f, ", {a:?}")?;
        }
        write!(f, "]")
    }
}

impl Eq for Sum {}
impl PartialEq for Sum {
    fn eq(&self, other: &Self) -> bool {
        if self.n_args() != other.n_args() {
            return false;
        }

        let mut largs = self.args.clone();
        largs.sort();
        let mut rargs = other.args.clone();
        rargs.sort();
        largs == rargs
    }
}

impl Sum {
    pub fn zero() -> Self {
        Self {
            args: Default::default(),
        }
    }

    pub fn is_zero(&self) -> bool {
        self.args.is_empty() || (self.args.len() == 1 && self.args[0].is_zero())
    }

    pub fn is_undef(&self) -> bool {
        self.args.first().is_some_and(|e| e.is_undef())
    }

    pub fn first(&self) -> Option<&Atom> {
        self.args.first().map(|a| a.atom())
    }

    pub fn as_binary_sum(&self) -> (Expr, Expr) {
        if self.args.is_empty() {
            (Expr::zero(), Expr::zero())
        } else if self.args.len() == 1 {
            (self.args[0].clone(), Expr::zero())
        } else {
            let a = self.args.first().unwrap().clone();
            let rest = &self.args[1..];
            let b = if rest.len() == 1 {
                rest[0].clone()
            } else {
                Expr::from(Atom::Sum(Sum {
                    args: rest.to_vec(),
                }))
            };
            (a, b)
        }
    }
}

#[derive(Clone, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct Prod {
    pub args: Vec<Expr>,
}

impl fmt::Debug for Prod {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.n_args() == 0 {
            return write!(f, "Prod[]");
        } else if self.n_args() == 1 {
            return write!(f, "Prod[{:?}]", self.args[0]);
        }
        let mut args = self.args.iter();

        if let Some(a) = args.next() {
            write!(f, "Prod[{a:?}")?;
        }

        for a in args {
            write!(f, ", {a:?}")?;
        }
        write!(f, "]")
    }
}

impl Eq for Prod {}
impl PartialEq for Prod {
    fn eq(&self, other: &Self) -> bool {
        if self.n_args() != other.n_args() {
            return false;
        }

        let mut largs = self.args.clone();
        largs.sort();
        let mut rargs = other.args.clone();
        rargs.sort();
        largs == rargs
    }
}

impl Prod {
    pub fn one() -> Self {
        Prod {
            args: Default::default(),
        }
    }

    pub fn is_one(&self) -> bool {
        self.args.is_empty() || (self.args.len() == 1 && self.args[0].is_one())
    }

    pub fn is_undef(&self) -> bool {
        self.args.first().is_some_and(|e| e.is_undef())
    }

    pub fn is_zero(&self) -> bool {
        self.args.first().is_some_and(|e| e.is_zero())
    }

    pub fn first(&self) -> Option<&Atom> {
        self.args.first().map(|a| a.atom())
    }

    pub fn as_binary_mul(&self) -> (Expr, Expr) {
        if self.args.is_empty() {
            (Expr::one(), Expr::one())
        } else if self.args.len() == 1 {
            (self.args[0].clone(), Expr::one())
        } else {
            let a = self.args.first().unwrap().clone();
            let rest = &self.args[1..];
            let b = if rest.len() == 1 {
                rest[0].clone()
            } else {
                Expr::from(Atom::Prod(Prod {
                    args: rest.to_vec(),
                }))
            };
            (a, b)
        }
    }

    pub fn term(&self) -> Option<Expr> {
        let mut terms: Vec<_> = self
            .iter_args()
            .filter(|a| !a.is_const())
            .cloned()
            .collect();

        if terms.is_empty() {
            None
        } else if terms.len() == 1 {
            Some(terms.remove(0))
        } else {
            Some(Self { args: terms }.into())
        }
    }
}

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct Pow {
    /// [base, exponent]
    pub(crate) args: [Expr; 2],
}

impl Pow {
    pub fn base(&self) -> &Expr {
        &self.args[0]
    }

    pub fn exponent(&self) -> &Expr {
        &self.args[1]
    }
}

impl fmt::Debug for Pow {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}^{:?}", self.args[0], self.args[1])
    }
}

#[derive(Clone, Default, Debug, Serialize, Deserialize)]
pub(crate) struct Explanation {
    pub(crate) explanation: PTR<str>,
    pub(crate) refs: Vec<Expr>,
}

const RECORD_STEPS: bool = false;

#[derive(Clone, Debug, Display, Serialize, Deserialize)]
#[debug("{:?}", self.atom())]
#[display("{}", self.fmt_ast())]
pub struct Expr {
    pub(crate) atom: PTR<Atom>,
    //pub(crate) expl: Option<Explanation>,
}

impl hash::Hash for Expr {
    fn hash<H: hash::Hasher>(&self, state: &mut H) {
        self.atom.hash(state);
    }
}

fn expr_as_cmp_slice<'a>(e: &'a Expr) -> &[Expr] {
    match e.atom() {
        Atom::Sum(_) | Atom::Prod(_) => e.args(),
        _ => slice::from_ref(e),
    }
}

impl cmp::Ord for Expr {
    fn cmp(&self, other: &Self) -> cmp::Ordering {
        self.atom().cmp(other.atom())
        /*
        if self.is_atom() || other.is_atom() {
            return self.atom().cmp(other.atom());
        }
        let lhs = expr_as_cmp_slice(self);
        let rhs = expr_as_cmp_slice(other);
        Self::cmp_slices(lhs, rhs)
        */
    }
}
impl cmp::PartialOrd for Expr {
    fn partial_cmp(&self, other: &Self) -> Option<cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl cmp::Eq for Expr {}
impl cmp::PartialEq for Expr {
    fn eq(&self, other: &Self) -> bool {
        self.atom == other.atom
    }
}

impl<T: Into<Atom>> From<T> for Expr {
    fn from(atom: T) -> Self {
        Self::from_atom(atom.into())
    }
}

impl ops::Deref for Expr {
    type Target = Atom;

    fn deref(&self) -> &Self::Target {
        self.atom.deref()
    }
}

macro_rules! func_atom {
    ($name: ident) => {
        pub fn $name(e: impl Borrow<Expr>) -> Expr {
            paste! {
                Expr::from(Atom::Func(Func::[<$name:camel>](e.borrow().clone())))
            }
        }
    };
}

impl Expr {
    //TODO store consts on stack
    std::thread_local! {
        static _UNDEF: Expr = Expr::from(Atom::UNDEF);

        static _MINUS_TWO: Expr = Expr::from(Atom::MINUS_TWO);
        static _MINUS_ONE: Expr = Expr::from(Atom::MINUS_ONE);
        static _ZERO: Expr = Expr::from(Atom::ZERO);
        static _ONE: Expr =  Expr::from(Atom::ONE);
        static _TWO: Expr = Expr::from(Atom::TWO);

        static _PI: Expr = Expr::from(Atom::PI);
        static _E: Expr = Expr::from(Atom::E);
    }

    fn from_atom(a: Atom) -> Expr {
        Self {
            atom: PTR::from(a),
            //expl: Default::default(),
        }
    }

    pub fn undef() -> Expr {
        Self::_UNDEF.with(|e| e.clone())
    }
    pub fn min_two() -> Expr {
        Self::_MINUS_TWO.with(|e| e.clone())
    }
    pub fn min_one() -> Expr {
        Self::_MINUS_ONE.with(|e| e.clone())
    }
    pub fn zero() -> Expr {
        Self::_ZERO.with(|e| e.clone())
    }
    pub fn one() -> Expr {
        Self::_ONE.with(|e| e.clone())
    }
    pub fn two() -> Expr {
        Self::_TWO.with(|e| e.clone())
    }
    pub fn pi() -> Expr {
        Self::_PI.with(|e| e.clone())
    }
    pub fn e() -> Expr {
        Self::_E.with(|e| e.clone())
    }

    /*
    pub fn min_two() -> Expr { Expr::from_atom(Atom::Rational(Rational::from(-2))) }
    pub fn min_one() -> Expr { Expr::from_atom(Atom::Rational(Rational::from(-1))) }
    pub fn zero() -> Expr { Expr::from_atom(Atom::Rational(Rational::from(0))) }
    pub fn one() -> Expr { Expr::from_atom(Atom::Rational(Rational::from(1))) }
    pub fn two() -> Expr { Expr::from_atom(Atom::Rational(Rational::from(2))) }
    */

    pub fn var(str: &str) -> Expr {
        Expr::from_atom(Atom::Var(str.into()))
    }

    pub fn atom(&self) -> &Atom {
        ops::Deref::deref(self)
    }
    pub fn atom_mut(&mut self) -> &mut Atom {
        PTR::make_mut(&mut self.atom)
    }

    pub fn rational<T: Into<Rational>>(r: T) -> Expr {
        Atom::Rational(r.into()).into()
    }

    func_atom!(cos);
    func_atom!(arc_cos);
    func_atom!(sin);
    func_atom!(arc_sin);
    func_atom!(tan);
    func_atom!(arc_tan);
    func_atom!(sec);
    func_atom!(arc_sec);
    func_atom!(cot);
    func_atom!(arc_cot);
    func_atom!(csc);
    func_atom!(arc_csc);

    pub fn exp(e: impl Borrow<Expr>) -> Expr {
        Expr::pow(Expr::e(), e)
    }
    pub fn log(base: impl Into<Real>, e: impl Borrow<Expr>) -> Expr {
        let base = base.into();
        Expr::from(Atom::Func(Func::Log(base, e.borrow().clone())))
    }
    pub fn ln(e: impl Borrow<Expr>) -> Expr {
        Expr::log(Irrational::E, e)
    }
    pub fn log10(e: impl Borrow<Expr>) -> Expr {
        Expr::log(Rational::from(10), e)
    }
    pub fn sqrt(v: impl Borrow<Expr>) -> Expr {
        let exp = Expr::from(Rational::from((1, 2)));
        Expr::pow(v, &exp)
    }

    #[inline(always)]
    pub fn as_monomial_view<'a>(&'a self, vars: &'a VarSet) -> MonomialView<'a> {
        MonomialView::new(self, vars)
    }
    #[inline(always)]
    pub fn as_polynomial_view<'a>(&'a self, vars: &'a VarSet) -> PolynomialView<'a> {
        PolynomialView::new(self, vars)
    }

    pub fn numerator(&self) -> Expr {
        use Atom as A;
        match self.atom() {
            A::Undef => self.clone(),
            A::Rational(r) => r.numer().into(),
            A::Pow(pow) => {
                if pow.exponent().is_min_one() {
                    Expr::one()
                } else {
                    self.clone()
                }
            }
            A::Prod(Prod { args }) => args
                .iter()
                .map(|a| a.numerator())
                .fold(Expr::one(), |prod, a| prod * a),
            _ => self.clone(),
        }
    }
    pub fn denominator(&self) -> Expr {
        use Atom as A;
        match self.atom() {
            A::Undef => self.clone(),
            A::Rational(r) => r.denom().into(),
            A::Pow(pow) => {
                if pow.exponent().is_min_one() {
                    pow.base().clone()
                } else {
                    Expr::one()
                }
            }
            A::Prod(Prod { args }) => args
                .iter()
                .map(|a| a.denominator())
                .fold(Expr::one(), |prod, a| prod * a),
            _ => Expr::one(),
        }
    }
    pub fn base(&self) -> Expr {
        match self.flatten().atom() {
            Atom::Pow(p) => p.base().clone(),
            _ => self.clone(),
        }
    }
    pub fn exponent(&self) -> Expr {
        match self.flatten().atom() {
            Atom::Pow(p) => p.exponent().clone(),
            _ => Expr::one(),
        }
    }
    pub fn is_exponential(&self) -> bool {
        self.is_pow() && self.base().is_e()
    }
    pub fn is_trig(&self) -> bool {
        self.try_unwrap_func_ref().is_ok_and(Func::is_trig)
    }

    pub fn try_as_div(&self) -> Option<(Expr, Expr)> {
        if self.is_pow() && self.exponent().is_min_one() {
            Some((Expr::min_one(), self.base().clone()))
        } else if self.is_prod() {
            Some(
                self.iter_args()
                    .map(|a| match a.exponent().is_neg() {
                        true => Err(Expr::pow(a.base(), Expr::min_one() * a.exponent())),
                        false => Ok(a),
                    })
                    .fold((Expr::one(), Expr::one()), |(n, d), rhs| match rhs {
                        Ok(numer) => (n * numer, d),
                        Err(denom) => (n, d * denom),
                    }),
            )
        } else {
            None
        }
    }

    //pub fn r#const(&self) -> Expr {
    //    match self.flatten().atom() {
    //        Atom::Var(_) | Atom::Sum(_) | Atom::Pow(_) | Atom::Func(_) => Expr::one(),
    //        Atom::Prod(prod) => prod
    //            .iter_args()
    //            .filter(|a| a.is_const())
    //            .cloned()
    //            .fold(Expr::one(), |lhs, rhs| lhs * rhs),
    //        Atom::Undef | Atom::Rational(_) | Atom::Irrational(_) => self.clone(),
    //    }
    //}

    pub fn rational_coeff(&self) -> Option<Rational> {
        if self.is_undef() {
            return None;
        }

        match self.atom() {
            Atom::Irrational(_) | Atom::Var(_) | Atom::Sum(_) | Atom::Pow(_) | Atom::Func(_) => {
                Some(Rational::ONE)
            }
            Atom::Prod(prod) => prod
                .iter_args()
                .filter_map(|a| a.try_unwrap_rational_ref().ok())
                .fold(Rational::ONE, |lhs, rhs| lhs * rhs)
                .into(),
            //Atom::Rational(r) => Some(r.clone()),
            Atom::Rational(_) | Atom::Undef => None,
        }
    }
    pub fn non_rational_term(&self) -> Option<Expr> {
        if self.is_undef() {
            return None;
        }

        match self.atom() {
            Atom::Irrational(_) | Atom::Var(_) | Atom::Sum(_) | Atom::Pow(_) | Atom::Func(_) => {
                Some(self.clone())
            }
            Atom::Prod(prod) => {
                let mut terms: Vec<_> = prod
                    .iter_args()
                    .filter(|a| !a.is_rational())
                    .cloned()
                    .collect();

                if terms.is_empty() {
                    None
                } else if terms.len() == 1 {
                    Some(terms.remove(0))
                } else {
                    Some(Prod { args: terms }.into())
                }
            }
            Atom::Undef | Atom::Rational(_) => None,
        }
    }

    /*
    pub fn term(&self) -> Option<Expr> {
        match self.flatten().atom() {
            Atom::Prod(prod) => {
                let mut terms: Vec<_> = prod
                    .iter_args()
                    .filter(|a| !a.is_const())
                    .cloned()
                    .collect();

                if terms.is_empty() {
                    None
                } else if terms.len() == 1 {
                    Some(terms.remove(0))
                } else {
                    Some(Prod { args: terms }.into())
                }
            }
            Atom::Var(_) | Atom::Sum(_) | Atom::Pow(_) | Atom::Func(_) => Some(self.clone()),

            Atom::Undef | Atom::Rational(_) | Atom::Irrational(_) => None,
        }
    }
    */

    pub fn variables(&self) -> HashSet<Expr> {
        let mut vars = Default::default();
        self.variables_impl(&mut vars);
        vars
    }
    fn variables_impl(&self, vars: &mut HashSet<Expr>) {
        use Atom as A;
        match self.atom() {
            A::Irrational(_) | A::Rational(_) | A::Undef => (),
            A::Var(_) => {
                vars.insert(self.clone());
            }
            A::Sum(Sum { args }) => args.iter().for_each(|a| a.variables_impl(vars)),
            A::Prod(Prod { args }) => {
                args.iter().for_each(|a| {
                    if let A::Sum(_) = a.atom() {
                        vars.insert(a.clone());
                    } else {
                        a.variables_impl(vars)
                    }
                });
            }
            A::Pow(pow) => {
                if let A::Rational(r) = pow.exponent().atom() {
                    if r >= &Rational::ONE {
                        vars.insert(pow.base().clone());
                        return;
                    }
                }
                vars.insert(self.clone());
            }
            A::Func(_) => todo!(),
        }
    }

    pub fn free_of(&self, expr: &Expr) -> bool {
        self.iter_compl_sub_exprs().all(|e| e != expr)
    }

    pub fn free_of_set<'a, I: IntoIterator<Item = &'a Self>>(&'a self, exprs: I) -> bool {
        exprs.into_iter().all(|e| self.free_of(e))
    }

    pub fn iter_compl_sub_exprs(&self) -> ExprIterator<'_> {
        let atoms = vec![self];
        ExprIterator { atoms }
    }

    pub fn flatten(&self) -> &Expr {
        match self.atom() {
            Atom::Sum(sum) if sum.n_args() == 1 => sum.args().first().unwrap(),
            Atom::Prod(prod) if prod.n_args() == 1 => prod.args().first().unwrap(),
            Atom::Pow(pow) if pow.exponent().is_one() => pow.base(),
            _ => self,
        }
    }

    fn cmp_slices(lhs: &[Self], rhs: &[Self]) -> cmp::Ordering {
        let iter = lhs.iter().zip(rhs.iter());
        for (l, r) in iter {
            let cmp = l.atom().cmp(r.atom());
            if !cmp.is_eq() {
                return cmp;
            }
        }

        if lhs.len() == rhs.len() {
            cmp::Ordering::Equal
        } else if lhs.len() < rhs.len() {
            cmp::Ordering::Greater
        } else {
            cmp::Ordering::Less
        }
    }

    fn cost(&self) -> usize {
        let mut cost = 1;
        self.iter_args().for_each(|e| cost += e.cost());
        cost
    }

    /*
    pub(crate) fn explain(mut self, explanation: impl AsRef<str>, refs: &[&Expr]) -> Self {
        if !RECORD_STEPS {
            return self;
        }

        let mut refs: Vec<_> = refs.iter().copied().cloned().collect();

        let mut expl_needed = false;
        for prev in &refs {
            if !(prev.is_atom() || prev == &self) {
                expl_needed = true;
            }
        }

        if !expl_needed {
            return self;
        }

        if self.expl.is_none() {
            self.expl = Some(Explanation {
                explanation: explanation.as_ref().into(),
                refs: refs.into(),
            });
            self
        } else {
            let mut next = self.clone();
            refs.push(self);
            next.expl = Some(Explanation {
                explanation: explanation.as_ref().into(),
                refs,
            });
            next
        }
    }

    pub fn clear_explanation(&mut self) {
        self.expl = None;
    }
    */
}

#[derive(Debug)]
pub struct ExprIterator<'a> {
    atoms: Vec<&'a Expr>,
}

impl<'a> Iterator for ExprIterator<'a> {
    type Item = &'a Expr;

    fn next(&mut self) -> Option<Self::Item> {
        self.atoms.pop().inspect(|expr| {
            expr.for_each_arg(|arg| self.atoms.push(arg));
        })
    }
}

pub trait SymbolicExpr: Clone + PartialOrd + PartialEq + Into<Expr> {
    /// basic simplification that can be applied regardless of context
    ///
    /// e.g 1 + 2 => 3
    fn reduce(&self) -> Expr {
        error!(
            "reduce called on irreducable type: {}",
            std::any::type_name_of_val(self)
        );
        self.clone().into()
    }

    fn is_atom(&self) -> bool {
        self.n_args() == 0
    }
    fn n_args(&self) -> usize {
        self.args().len()
    }
    fn args(&self) -> &[Expr] {
        &[]
    }
    fn args_mut(&mut self) -> &mut [Expr] {
        &mut []
    }
    fn iter_args(&self) -> impl Iterator<Item = &Expr> {
        self.args().iter()
    }
    //fn map_args(&mut self, func: impl FnMut(&mut Expr)) {
    //    self.args_mut().iter_mut().for_each(func)
    //}
    fn map_args(mut self, map_fn: impl Fn(&mut Expr)) -> Self {
        self.args_mut().iter_mut().for_each(map_fn);
        self
    }
}

impl SymbolicExpr for Atom {
    fn args(&self) -> &[Expr] {
        use Atom as A;
        match self {
            A::Undef | A::Rational(_) | A::Irrational(_) | A::Var(_) => &[],
            A::Sum(sum) => sum.args(),
            A::Prod(prod) => prod.args(),
            A::Pow(pow) => pow.args(),
            A::Func(func) => func.args(),
        }
    }

    fn args_mut(&mut self) -> &mut [Expr] {
        use Atom as A;
        match self {
            A::Undef | A::Rational(_) | A::Irrational(_) | A::Var(_) => &mut [],
            A::Sum(sum) => sum.args_mut(),
            A::Prod(prod) => prod.args_mut(),
            A::Pow(pow) => pow.args_mut(),
            A::Func(func) => func.args_mut(),
        }
    }
}

impl SymbolicExpr for Irrational {}
impl SymbolicExpr for Rational {}
impl SymbolicExpr for Var {}

impl SymbolicExpr for Expr {
    fn reduce(&self) -> Self {
        use Atom as A;
        let res = self.clone().map_args(|a| *a = a.reduce());
        match res.atom() {
            A::Irrational(_) | A::Undef | A::Rational(_) | A::Var(_) => res,
            A::Sum(sum) => sum.reduce(),
            A::Prod(prod) => prod.reduce(),
            A::Pow(pow) => pow.reduce(),
            A::Func(func) => func.reduce(),
        }
    }

    fn args(&self) -> &[Expr] {
        self.atom().args()
    }
    fn args_mut(&mut self) -> &mut [Expr] {
        self.atom_mut().args_mut()
    }
}

impl SymbolicExpr for Sum {
    fn reduce(&self) -> Expr {
        let mut sum = Sum::reduce_rec(&self.args);
        if sum.is_zero() {
            Expr::zero()
        } else if sum.is_undef() {
            Expr::undef()
        } else if sum.args.len() == 1 {
            sum.args.remove(0)
        } else {
            Atom::Sum(sum).into()
        }
    }

    fn args(&self) -> &[Expr] {
        self.args.as_slice()
    }
    fn args_mut(&mut self) -> &mut [Expr] {
        self.args.as_mut_slice()
    }
}

impl SymbolicExpr for Prod {
    fn reduce(&self) -> Expr {
        let mut prod = Prod::reduce_rec(&self.args);
        if prod.is_one() {
            Expr::one()
        } else if prod.is_undef() {
            Expr::undef()
        } else if prod.args.len() == 1 {
            prod.args.remove(0)
        } else {
            Atom::Prod(prod).into()
        }
    }

    fn args(&self) -> &[Expr] {
        self.args.as_slice()
    }
    fn args_mut(&mut self) -> &mut [Expr] {
        self.args.as_mut_slice()
    }
}

impl SymbolicExpr for Pow {
    fn reduce(&self) -> Expr {
        use Atom as A;
        if self.base().is_undef()
            || self.exponent().is_undef()
            || (self.base().is_zero() && self.exponent().is_zero())
            || (self.base().is_zero() && self.exponent().is_neg())
        {
            return Expr::undef();
        }

        if self.base().is_one() {
            return Expr::one();
        } else if self.exponent().is_one() {
            return self.base().clone();
        } else if self.exponent().is_zero() {
            return Expr::one();
        }

        match (self.base().atom(), self.exponent().atom()) {
            (A::Rational(b), A::Rational(e)) => {
                let (res, rem) = b.clone().pow(e.clone());
                if rem.is_zero() {
                    Expr::from(res)
                } else {
                    //Expr::from(res) * Expr::from(b.clone()).pow(Expr::from(rem))
                    A::Pow(self.clone()).into()
                }
            }
            //TODO: rule?
            (A::Pow(pow), _) => {
                let mut pow = pow.clone();
                pow.args[1] *= self.exponent();
                pow.reduce()
            }
            _ => A::Pow(self.clone()).into(),
        }
    }

    fn args(&self) -> &[Expr] {
        &self.args
    }
    fn args_mut(&mut self) -> &mut [Expr] {
        &mut self.args
    }
}

impl SymbolicExpr for Func {
    fn reduce(&self) -> Expr {
        use Func as F;

        //let mut e = self.clone();
        let e = self.clone().map_args(|a| *a = a.reduce());
        match e {
            F::Sin(x) => {
                if x.is_zero() {
                    return Expr::zero();
                }

                if let Some(c) = x.rational_coeff() {
                    if c.is_min_one() {
                        return Expr::min_one() * Expr::sin(x.non_rational_term().unwrap());
                    } else if c.is_neg() {
                        return Expr::min_one() * Expr::sin(Expr::min_one() * x);
                    }
                }

                Expr::sin(x)
            }

            F::Cos(x) => {
                if x.is_zero() {
                    return Expr::one();
                }

                if let Some(c) = x.rational_coeff() {
                    if c.is_min_one() {
                        return Expr::cos(x.non_rational_term().unwrap());
                    } else if c.is_neg() {
                        return Expr::cos(Expr::min_one() * x);
                    }
                }

                Expr::cos(x)
            }
            // sin(pi/6) => 1/2
            // sin(pi/4) => 1/sqrt(2)
            // sin(pi/3) => sqrt(3)/2
            //F::Sin(x) if x.numerator().is_pi() => {
            //    todo!()
            //}
            F::Log(base, x) if x.atom() == &base => Expr::one(),
            _ => e.into(),
        }
    }

    fn args_mut(&mut self) -> &mut [Expr] {
        use Func as F;
        match self {
            F::Sin(x)
            | F::Cos(x)
            | F::Tan(x)
            | F::Sec(x)
            | F::Cot(x)
            | F::Csc(x)
            | F::ArcSin(x)
            | F::ArcCos(x)
            | F::ArcTan(x)
            | F::ArcSec(x)
            | F::ArcCot(x)
            | F::ArcCsc(x)
            | F::Log(_, x) => slice::from_mut(x),
        }
    }

    fn args(&self) -> &[Expr] {
        use Func as F;
        match self {
            F::Sin(x)
            | F::Cos(x)
            | F::Tan(x)
            | F::Sec(x)
            | F::Cot(x)
            | F::Csc(x)
            | F::ArcSin(x)
            | F::ArcCos(x)
            | F::ArcTan(x)
            | F::ArcSec(x)
            | F::ArcCot(x)
            | F::ArcCsc(x)
            | F::Log(_, x) => slice::from_ref(x),
        }
    }
}

pub mod unicode {
    use paste::paste;

    macro_rules! symbl {
        ($id:ident: $unicode: literal) => {
            paste! {
                pub const fn [<$id:snake>]() -> &'static str {
                    $unicode
                }
            }
        };
    }

    symbl!(pi: "π");
    symbl!(e : "𝓮");
    symbl!(sub : "−");
    symbl!(unry_sub : "-");
    symbl!(add : "+");
    symbl!(mul : "·");
    symbl!(frac_slash : "/");
    symbl!(pow : "^");
    symbl!(undef : "∅");
}

#[cfg(test)]
mod test {
    use super::*;
    use assert_eq as eq;
    use calcurs_macros::expr as e;

    #[test]
    fn variables() {
        eq!(
            e!(x ^ 3 + 3 * x ^ 2 * y + 3 * x * y ^ 2 + y ^ 3).variables(),
            [e!(x), e!(y)].into_iter().collect()
        );
        eq!(
            e!(3 * x * (x + 1) * y ^ 2 * z ^ n).variables(),
            [e!(x), e!(x + 1), e!(y), e!(z ^ n)].into_iter().collect()
        );
        eq!(
            e!(2 ^ (1 / 2) * x ^ 2 + 3 ^ (1 / 2) * x + 5 ^ (1 / 2)).variables(),
            [e!(x), e!(2 ^ (1 / 2)), e!(3 ^ (1 / 2)), e!(5 ^ (1 / 2))]
                .into_iter()
                .collect()
        );
    }

    #[test]
    fn num_denom() {
        let nd = |e: Expr| (e.numerator(), e.denominator());
        eq!(
            nd(e!((2 / 3) * (x * (x + 1)) / (x + 2) * y ^ n)),
            (e!(2 * x * (x + 1) * y ^ n), e!(3 * (x + 2)))
        );
    }

    #[test]
    fn derivative() {
        let d = |e: Expr| {
            e.derivative(e!(x))
                .rationalize()
                .expand()
                .factor_out()
                .reduce()
        };

        eq!(d(e!(x ^ 2)), e!(2 * x));
        eq!(d(e!(sin(x))), e!(cos(x)));
        eq!(d(e!(exp(x))), e!(exp(x)));
        eq!(d(e!(x * exp(x))), e!(exp(x) * (1 + x)));
        eq!(d(e!(ln(x))), e!(1 / x));
        eq!(d(e!(1 / x)), e!(-1 / x ^ 2));
        eq!(d(e!(tan(x))), e!(sec(x) ^ 2));
        eq!(d(e!(arc_tan(x))), e!(1 / (x ^ 2 + 1)));
        eq!(
            d(e!(x * ln(x) * sin(x))),
            e!(x * cos(x) * ln(x) + sin(x) * ln(x) + sin(x)).sort_args() //e!(sin(x) + x*cos(x)*ln(x) + sin(x)*ln(x))
        );
        eq!(d(e!(x ^ 2)), e!(2 * x));
        //eq!(d(exp(e!(sin(x)))), exp(e!(x)));
    }

    #[test]
    fn term_const() {
        eq!(e!(2 * y).non_rational_term(), Some(e!(y)));
        eq!(e!(x * y).non_rational_term(), Some(e!(x * y)));
        eq!(e!(x).rational_coeff(), Some(Rational::ONE));
        eq!(e!(2 * x).rational_coeff(), Some(Rational::TWO));
        eq!(e!(y * x).rational_coeff(), Some(Rational::ONE));
    }
}
