use crate::{
    fmt_ast,
    polynomial::{MonomialView, PolynomialView, VarSet},
    rational::{Int, Rational},
    utils::{self, HashSet},
};
use std::{borrow::Borrow, fmt, ops, slice};

use derive_more::{Debug, Display, From, Into, IsVariant, TryUnwrap, Unwrap};
use paste::paste;

pub(crate) type PTR<T> = std::rc::Rc<T>;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct Derivative {
    arg: Expr,
    var: Expr,
    degree: u64,
}

#[derive(
    Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, From, IsVariant, Unwrap, TryUnwrap,
)]
#[unwrap(ref)]
#[try_unwrap(ref)]
pub enum Atom {
    #[debug("undef")]
    Undef,
    #[from(i32, i64, u32, u64, Int, Rational)]
    Rational(Rational),
    #[from]
    Irrational(Irrational),
    #[from(&str)]
    Var(Var),
    Sum(Sum),
    Prod(Prod),
    Pow(Pow),
    Func(Func),
}

impl Atom {
    pub const UNDEF: Atom = Atom::Undef;
    pub const MINUS_ONE: Atom = Atom::Rational(Rational::MINUS_ONE);
    pub const ZERO: Atom = Atom::Rational(Rational::ZERO);
    pub const ONE: Atom = Atom::Rational(Rational::ONE);

    pub fn is_zero(&self) -> bool {
        self == &Atom::ZERO
    }
    pub fn is_one(&self) -> bool {
        self == &Atom::ONE
    }
    pub fn is_min_one(&self) -> bool {
        self == &Atom::MINUS_ONE
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
    pub fn is_atom(&self) -> bool {
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
        self.is_undef() || self.is_number()
    }

    pub fn args(&self) -> &[Expr] {
        use Atom as A;
        match self {
            A::Undef | A::Rational(_) | A::Irrational(_) | A::Var(_) => &[],
            A::Sum(Sum { args }) | A::Prod(Prod { args }) => args.as_slice(),
            A::Pow(Pow { args }) => args,
            A::Func(func) => func.args(),
        }
    }

    pub fn args_mut(&mut self) -> &mut [Expr] {
        use Atom as A;
        match self {
            A::Undef | A::Rational(_) | A::Irrational(_) | A::Var(_) => &mut [],
            A::Sum(Sum { args }) | A::Prod(Prod { args }) => args.as_mut_slice(),
            A::Pow(Pow { args }) => args,
            A::Func(func) => func.args_mut(),
        }
    }

    pub fn for_each_arg<'a>(&'a self, func: impl FnMut(&'a Expr)) {
        self.args().iter().for_each(func)
    }
}

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Display, Debug, From)]
#[from(&str)]
pub struct Var(pub(crate) PTR<str>);
impl Var {
    fn len(&self) -> usize {
        self.0.len()
    }

    fn as_ref(&self) -> &str {
        self.0.as_ref()
    }
}

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Display, From)]
#[from(forward)]
pub enum Real {
    #[debug("{_0}")]
    Rational(Rational),
    #[debug("{_0}")]
    Irrational(Irrational),
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

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Display)]
pub enum Irrational {
    #[debug("e")]
    #[display("𝓮")]
    E,
    #[debug("pi")]
    #[display("π")]
    PI,
}

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Display)]
#[display("{}{_0}", self.name())]
pub enum Func {
    Sin(Expr),
    Cos(Expr),
    Tan(Expr),
    Sec(Expr),
    ArcSin(Expr),
    ArcCos(Expr),
    ArcTan(Expr),
    Exp(Expr),
    #[display("{}{_1}", self.name())]
    Log(Real, Expr),
}

impl Func {
    pub fn is_nat_log(&self) -> bool {
        match self {
            Func::Log(base, _) => base == &Real::from(Irrational::E),
            _ => false,
        }
    }

    pub fn name(&self) -> String {
        match self {
            Func::Sin(_) => "sin".into(),
            Func::Cos(_) => "cos".into(),
            Func::Tan(_) => "tan".into(),
            Func::Sec(_) => "sec".into(),
            Func::ArcSin(_) => "arcsin".into(),
            Func::ArcCos(_) => "arccos".into(),
            Func::ArcTan(_) => "arctan".into(),
            Func::Exp(_) => "exp".into(),
            Func::Log(Real::Irrational(Irrational::E), _) => "ln".into(),
            Func::Log(Real::Rational(r), _) if r == &Rational::from(10) => "log".into(),
            Func::Log(base, _) => format!("log{base}"),
        }
    }

    pub fn args(&self) -> &[Expr] {
        use Func as F;
        match self {
            F::Sin(x)
            | F::Cos(x)
            | F::Tan(x)
            | F::Sec(x)
            | F::ArcSin(x)
            | F::ArcCos(x)
            | F::ArcTan(x)
            | F::Exp(x)
            | F::Log(_, x) => slice::from_ref(x),
        }
    }

    pub fn args_mut(&mut self) -> &mut [Expr] {
        use Func as F;
        match self {
            F::Sin(x)
            | F::Cos(x)
            | F::Tan(x)
            | F::Sec(x)
            | F::ArcSin(x)
            | F::ArcCos(x)
            | F::ArcTan(x)
            | F::Exp(x)
            | F::Log(_, x) => slice::from_mut(x),
        }
    }

    pub fn iter_args(&self) -> impl Iterator<Item = &Expr> {
        self.args().iter()
    }

    pub fn derivative(&self, x: impl Borrow<Expr>) -> Expr {
        use Expr as E;
        use Func as F;

        let r = |n: i32, d: i32| Expr::from(Rational::from((n, d)));
        let e = |e: i32| Expr::from(e);
        let d = |e: &Expr| -> Expr { e.derivative(x) };

        match self {
            F::Sin(f) => d(f) * E::cos(f),
            F::Cos(f) => d(f) * e(-1) * E::sin(f),
            F::Tan(f) => d(f) * E::pow(E::sec(f), &e(2)),
            F::Sec(f) => d(f) * E::tan(f) * E::sec(f),
            F::ArcSin(f) => d(f) * E::pow(e(1) - E::pow(f, e(2)), r(-1, 2)),
            F::ArcCos(f) => d(f) * e(-1) * E::pow(e(1) - E::pow(f, e(2)), r(-1, 2)),
            F::ArcTan(f) => d(f) * E::pow(e(1) + E::pow(f, e(2)), e(-1)),
            F::Log(base, f) => d(f) * Expr::pow(f * E::ln(Expr::from(base.clone())), e(-1)),
            F::Exp(f) => Expr::exp(f) * d(f),
        }
    }
    //pub fn iter_args_mut(&mut self) -> impl Iterator<Item = &mut Expr> {
    //    self.args_mut().iter_mut()
    //}

    pub fn reduce(&self) -> Expr {
        use Func as F;
        match self {
            F::Log(base, x) if x.atom() == base => Expr::one(),
            _ => Atom::Func(self.clone()).into(),
        }
    }
}

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
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
                    args: rest.into_iter().cloned().collect(),
                }))
            };
            (a, b)
        }
    }

    pub fn reduce(&self) -> Expr {
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
}

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

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Prod {
    pub args: Vec<Expr>,
}
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
                    args: rest.into_iter().cloned().collect(),
                }))
            };
            (a, b)
        }
    }

    pub fn reduce(&self) -> Expr {
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
}

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
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

    pub fn reduce(&self) -> Expr {
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
}

impl fmt::Debug for Pow {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}^{:?}", self.args[0], self.args[1])
    }
}

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Display)]
#[display("{}", fmt_ast::FmtAtom::from(self.atom()))]
pub struct Expr(PTR<Atom>);

impl<T: Into<Atom>> From<T> for Expr {
    fn from(value: T) -> Self {
        Expr(PTR::new(value.into()))
    }
}

impl ops::Deref for Expr {
    type Target = Atom;

    fn deref(&self) -> &Self::Target {
        self.0.deref()
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
    pub fn undef() -> Expr {
        Atom::Undef.into()
    }

    pub fn pi() -> Expr {
        Irrational::PI.into()
    }

    pub fn zero() -> Expr {
        Atom::ZERO.into()
    }

    pub fn one() -> Expr {
        Atom::ONE.into()
    }

    pub fn min_one() -> Expr {
        Atom::MINUS_ONE.into()
    }

    pub fn var(str: &str) -> Expr {
        Expr::from(str)
    }

    pub fn atom(&self) -> &Atom {
        ops::Deref::deref(self)
    }
    pub fn atom_mut(&mut self) -> &mut Atom {
        PTR::make_mut(&mut self.0)
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
    func_atom!(exp);

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
        Expr::pow(v, &Rational::from((1, 2)).into())
    }

    #[inline(always)]
    pub fn as_monomial<'a>(&'a self, vars: &'a VarSet) -> MonomialView<'a> {
        MonomialView::new(self, vars)
    }
    #[inline(always)]
    pub fn as_polynomial<'a>(&'a self, vars: &'a VarSet) -> PolynomialView<'a> {
        PolynomialView::new(self, vars)
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
        match self.get_flatten() {
            Atom::Pow(p) => p.base().clone(),
            _ => self.clone(),
        }
    }
    pub fn exponent(&self) -> Expr {
        match self.get_flatten() {
            Atom::Pow(p) => p.exponent().clone(),
            _ => Expr::one(),
        }
    }

    pub fn r#const(&self) -> Option<Expr> {
        match self.atom() {
            Atom::Var(_) | Atom::Sum(_) | Atom::Pow(_) | Atom::Func(_) => Some(Expr::one()),
            Atom::Undef | Atom::Rational(_) | Atom::Irrational(_) => None,
            Atom::Prod(Prod { args }) => {
                if let Some(a) = args.first() {
                    if a.is_const() {
                        return Some(a.clone());
                    }
                }
                Some(Expr::one())
            }
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

    pub fn get_flatten(&self) -> &Atom {
        match self.atom() {
            Atom::Sum(Sum { args }) | Atom::Prod(Prod { args }) if args.len() == 1 => {
                args.first().unwrap().atom()
            }
            a => a,
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

    pub fn reduce(&self) -> Self {
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

    pub fn iter_compl_sub_exprs(&self) -> ExprIterator<'_> {
        let atoms = vec![self];
        ExprIterator { atoms }
    }
}

#[derive(Debug)]
pub struct ExprIterator<'a> {
    atoms: Vec<&'a Expr>,
}

impl<'a> Iterator for ExprIterator<'a> {
    type Item = &'a Expr;

    fn next(&mut self) -> Option<Self::Item> {
        self.atoms.pop().map(|expr| {
            expr.for_each_arg(|arg| self.atoms.push(arg));
            expr
        })
    }
}
