use crate::{
    fmt_ast,
    polynomial::{MonomialView, PolynomialView, VarSet},
    rational::{Int, Rational},
    transforms::Step,
    utils::{self, log_macros::*, HashSet},
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

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Display, Debug, From, Serialize, Deserialize)]
#[from(&str, String)]
pub struct Var(pub(crate) PTR<str>);

#[derive(
    Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, From, IsVariant, Unwrap, TryUnwrap, Serialize, Deserialize
)]
#[unwrap(ref)]
#[try_unwrap(ref)]
pub enum Atom {
    #[debug("undef")]
    Undef,
    #[from(i32, i64, u32, u64, Int, Rational)]
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
            Atom::Undef
            | Atom::Rational(_)
            | Atom::Irrational(_)
            | Atom::Var(_)
            | Atom::Func(_) => true,

            Atom::Sum(_) | Atom::Prod(_) | Atom::Pow(_) => false,
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

    pub fn try_unwrap_int(&self) -> Option<Int> {
        match self {
            Atom::Rational(r) if r.is_int() && r.is_neg() => Some(Int::MINUS_ONE * r.numer()),
            Atom::Rational(r) if r.is_int() => Some(r.numer()),
            _ => None,
        }
    }
    pub fn unwrap_int(&self) -> Int {
        self.try_unwrap_int().unwrap()
    }

    pub fn for_each_arg<'a>(&'a self, func: impl FnMut(&'a Expr)) {
        self.args().iter().for_each(func)
    }
}

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Display, From, Serialize, Deserialize)]
#[from(forward)]
pub enum Real {
    #[debug("{_0}")]
    Rational(Rational),
    #[debug("{_0}")]
    Irrational(Irrational),
}

impl Real {
    pub const E: Real = Real::Irrational(Irrational::E);
    pub const PI: Real = Real::Irrational(Irrational::PI);
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

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Display, Unwrap, TryUnwrap, Serialize, Deserialize)]
#[unwrap(ref)]
#[try_unwrap(ref)]
#[display("{}({_0})", self.name())]
pub enum Func {
    Sin(Expr),
    Cos(Expr),
    Tan(Expr),
    Sec(Expr),
    ArcSin(Expr),
    ArcCos(Expr),
    ArcTan(Expr),
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
            | Func::Cos(_)
            | Func::Tan(_)
            | Func::Sec(_)
            | Func::ArcSin(_)
            | Func::ArcCos(_)
            | Func::ArcTan(_) => true,
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
            Func::Cos(_) => "cos",
            Func::Tan(_) => "tan",
            Func::Sec(_) => "sec",
            Func::ArcSin(_) => "arcsin",
            Func::ArcCos(_) => "arccos",
            Func::ArcTan(_) => "arctan",
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
            F::Log(base, f) => d(f) * E::pow(f * E::ln(E::from(base.clone())), e(-1)),
            //F::Exp(f) => E::exp(f) * d(f),
        }
    }
}

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[cfg_attr(feature = "default_debug", derive(Debug))]
pub struct Sum {
    pub args: Vec<Expr>,
}

impl Sum {
    pub fn zero() -> Self {
        Self {
            args: Default::default(),
        }
    }

    pub fn is_zero(&self) -> bool {
        self.args.is_empty()
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

#[cfg(not(feature = "default_debug"))]
impl fmt::Debug for Sum {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.args.is_empty() {
            return write!(f, "(+)");
        } else if self.args.len() == 1 {
            return write!(f, "(+{:?})", self.args[0]);
        }
        utils::fmt_iter(
            ["(", " + ", ")"],
            self.args.iter(),
            |a, f| write!(f, "{a:?}"),
            f,
        )
    }
}

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[cfg_attr(feature = "default_debug", derive(Debug))]
pub struct Prod {
    pub args: Vec<Expr>,
}
#[cfg(not(feature = "default_debug"))]
impl fmt::Debug for Prod {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.args.is_empty() {
            return write!(f, "(*)");
        } else if self.args.len() == 1 {
            return write!(f, "(1*{:?})", self.args[0]);
        }
        utils::fmt_iter(
            ["(", " * ", ")"],
            self.args.iter(),
            |a, f| write!(f, "{a:?}"),
            f,
        )
    }
}

impl Prod {
    pub fn one() -> Self {
        Prod {
            args: Default::default(),
        }
    }

    pub fn is_one(&self) -> bool {
        self.args.is_empty()
    }

    pub fn is_undef(&self) -> bool {
        self.args.first().is_some_and(|e| e.is_undef())
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
}

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[cfg_attr(feature = "default_debug", derive(Debug))]
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

#[cfg(not(feature = "default_debug"))]
impl fmt::Debug for Pow {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}^{:?}", self.args[0], self.args[1])
    }
}

#[derive(Clone, Debug, Display, Serialize, Deserialize)]
#[debug("{:?}", self.atom())]
#[display("{}", fmt_ast::FmtAtom::from(self.atom()))]
pub struct Expr {
    pub(crate) atom: PTR<Atom>,
    pub(crate) expls: Vec<Step>,
}

impl hash::Hash for Expr {
    fn hash<H: hash::Hasher>(&self, state: &mut H) {
        // Only hash the `id` field
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
        if self.is_atom() || other.is_atom() {
            return self.atom().cmp(other.atom());
        }
        let lhs = expr_as_cmp_slice(self);
        let rhs = expr_as_cmp_slice(other);
        Self::cmp_slices(lhs, rhs)
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
        Self {
            atom: PTR::from(atom.into()),
            expls: vec![],
        }
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
            expls: vec![],
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
    func_atom!(sin);
    func_atom!(tan);
    func_atom!(sec);
    func_atom!(arc_sin);
    func_atom!(arc_cos);
    func_atom!(arc_tan);

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

    pub fn explain(mut self, from: Option<Self>, explanation: impl AsRef<str>) -> Self {
        if Some(&self) == from.as_ref() {
            self
        } else {
            let step = Step {
                to: self.clone(),
                from,
                explanation: explanation.as_ref().to_string(),
            };
            self.expls.push(step);
            self
        }
    }

    pub fn steps(&self) -> &Vec<Step> {
        &self.expls
    }

    pub fn show_steps(&self) {
        println!("{}:", self);
        println!("{:?}", self.steps());
        self.iter_args().for_each(Self::show_steps);
    }

    pub fn numerator(&self) -> Expr {
        use Atom as A;
        match self.atom() {
            A::Undef => self.clone(),
            A::Rational(r) => r.numer().into(),
            A::Pow(pow) => match pow.exponent().atom() {
                &A::MINUS_ONE => Expr::one(),
                _ => self.clone(),
            },
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
            A::Pow(pow) => match pow.exponent().atom() {
                &A::MINUS_ONE => pow.base().clone(),
                _ => Expr::one(),
            },
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

    pub fn r#const(&self) -> Option<Expr> {
        match self.flatten().atom() {
            Atom::Var(_) | Atom::Sum(_) | Atom::Pow(_) | Atom::Func(_) => Some(Expr::one()),
            Atom::Prod(Prod { args }) => {
                if let Some(a) = args.first() {
                    if a.is_const() {
                        return Some(a.clone());
                    }
                }
                Some(Expr::one())
            }

            Atom::Undef | Atom::Rational(_) | Atom::Irrational(_) => None,
        }
    }

    pub fn term(&self) -> Option<Expr> {
        match self.atom() {
            Atom::Prod(prod) => {
                if let Some(a) = prod.args.first() {
                    if a.is_const() {
                        let (_, c) = prod.as_binary_mul();
                        return Some(c);
                    }
                }
                Some(self.clone())
            }
            Atom::Var(_) | Atom::Sum(_) | Atom::Pow(_) | Atom::Func(_) => Some(self.clone()),

            Atom::Undef | Atom::Rational(_) | Atom::Irrational(_) => None,
        }
    }

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
        match (self.base().atom(), self.exponent().atom()) {
            (A::Undef, _) | (_, A::Undef) | (&A::ZERO, &A::ZERO) => Expr::undef(),
            (&A::ZERO, A::Rational(r)) if r.is_neg() => Expr::undef(),
            (&A::ZERO, A::Rational(_)) => Expr::zero(),
            (&A::ONE, _) => Expr::one(),
            (_, &A::ONE) => self.base().clone(),
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
            // sin(0) => 0
            F::Sin(x) if x.is_zero() => x.clone(),

            F::Sin(x)
                if x.is_prod()
                    && x.n_args() != 0
                    && x.args()[0].is_rational_and(|r| r.is_int() && r.is_neg()) =>
            {
                let (n, x) = x.atom().clone().unwrap_prod().as_binary_mul();
                debug_assert!(n.is_int() && n.is_neg());
                let n = n.atom().unwrap_int();
                Expr::min_one() * Expr::sin(Expr::from(n.abs()) * x)
            }

            F::Cos(x)
                if x.is_prod()
                    && x.n_args() != 0
                    && x.args()[0].is_rational_and(|r| r.is_int() && r.is_neg()) =>
            {
                let (n, x) = x.atom().clone().unwrap_prod().as_binary_mul();
                debug_assert!(n.is_int() && n.is_neg());
                let n = n.atom().unwrap_int();
                Expr::cos(Expr::from(n.abs()) * x)
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
            | F::ArcSin(x)
            | F::ArcCos(x)
            | F::ArcTan(x)
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
            | F::ArcSin(x)
            | F::ArcCos(x)
            | F::ArcTan(x)
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
        eq!(d(e!(x * exp(x))), e!((x + 1) * exp(x)));
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
        eq!(e!(2 * y).term(), Some(e!(y)));
        eq!(e!(x * y).term(), Some(e!(x * y)));
        eq!(e!(x).r#const(), Some(e!(1)));
        eq!(e!(2 * x).r#const(), Some(e!(2)));
        eq!(e!(y * x).r#const(), Some(e!(1)));
    }
}
