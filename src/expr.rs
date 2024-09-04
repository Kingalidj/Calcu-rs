use std::{borrow::Borrow, collections::VecDeque, fmt, iter, ops, rc::Rc};

use crate::{
    fmt_ast::{self, FmtAst},
    polynomial::{MonomialView, PolynomialView, VarSet},
    rational::{Rational, UInt},
    utils::{self, trace_fn, HashMap, HashSet, Pow as Power},
};

impl From<Atom> for Expr {
    fn from(value: Atom) -> Self {
        Expr(value.into())
    }
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub struct Sum {
    pub(crate) args: VecDeque<Expr>,
}
impl Sum {
    pub fn zero() -> Self {
        Self {
            args: Default::default(),
        }
    }

    fn first(&self) -> Option<&Atom> {
        self.args.front().map(|a| a.get())
    }

    pub fn add_rhs(&mut self, rhs: &Expr) {
        use Atom as A;

        if let Some(A::Undef) = self.first() {
            return;
        }

        match rhs.get() {
            &A::ZERO => (),
            A::Undef => {
                self.args.clear();
                self.args.push_back(rhs.clone())
            }
            A::Sum(sum) => {
                sum.args.iter().for_each(|a| self.add_rhs(a));
            }
            // sum all rationals in the first element
            A::Rational(r) => {
                if let Some(A::Rational(r2)) = self.first() {
                    self.args[0] = Atom::Rational(r.clone() + r2).into();
                } else {
                    self.args.push_front(rhs.clone())
                }
            }
            A::Var(_) | A::Prod(_) | A::Pow(_) => self.args.push_back(rhs.clone()),
        }
    }

    fn reduce(&self) -> Expr {
        use Atom as A;
        if self.args.is_empty() {
            return Expr::zero();
        } else if self.args.len() == 1 {
            return self.args.front().unwrap().clone();
        }

        let mut res = Sum::zero();
        //res.args.push_back(Expr::zero());

        let mut accum = Rational::ZERO;

        for a in &self.args {
            match a.get() {
                A::Var(_) | A::Sum(_) | A::Prod(_) | A::Pow(_) => res.add_rhs(a),
                A::Undef => return Expr::undef(),
                A::Rational(r) => accum += r,
            }
        }

        //res.args[0] = accum.into();
        if !accum.is_zero() {
            res.args.push_front(accum.into());
        }

        Atom::Sum(res).into()
    }
}
impl fmt::Debug for Sum {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.args.is_empty() {
            return write!(f, "(+)");
        }
        utils::fmt_iter(
            ["(", " + ", ")"],
            self.args.iter(),
            |a, f| write!(f, "{:?}", a),
            f,
        )
    }
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub struct Prod {
    pub(crate) args: VecDeque<Expr>,
}
impl fmt::Debug for Prod {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.args.is_empty() {
            return write!(f, "(*)");
        }
        utils::fmt_iter(
            ["(", " * ", ")"],
            self.args.iter(),
            |a, f| write!(f, "{:?}", a),
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

    fn first(&self) -> Option<&Atom> {
        self.args.front().map(|a| a.get())
    }

    pub fn mul_rhs(&mut self, rhs: &Expr) {
        use Atom as A;

        if let Atom::Undef = rhs.get() {
            self.args.clear();
            self.args.push_back(rhs.clone());
            return;
        }

        match self.first() {
            Some(A::Undef | &A::ZERO) => return,
            Some(_) | None => (),
        }

        match rhs.get() {
            &A::ONE => (),
            A::Undef | &A::ZERO => {
                self.args.clear();
                self.args.push_back(rhs.clone())
            }
            A::Prod(prod) => {
                prod.args.iter().for_each(|a| self.mul_rhs(a));
            }
            A::Rational(r) => {
                if let Some(A::Rational(r2)) = self.first() {
                    self.args[0] = A::Rational(r.clone() * r2).into();
                } else {
                    self.args.push_front(rhs.clone())
                }
            }
            A::Var(_) | A::Sum(_) | A::Pow(_) => self.args.push_back(rhs.clone()),
        }
    }

    fn expand_binary_mul(lhs: &Expr, rhs: &Expr) -> Expr {
        use Atom as A;
        match (lhs.get(), rhs.get()) {
            (A::Prod(_), A::Prod(_)) => {
                let mut res = Prod::one();
                res.mul_rhs(lhs);
                res.mul_rhs(rhs);
                A::Prod(res).into()
            }
            (_, A::Sum(sum)) => {
                let mut res = Sum::zero();
                for term in &sum.args {
                    let term_prod = Self::expand_binary_mul(lhs, term);
                    res.add_rhs(&term_prod)
                }
                A::Sum(res).into()
            }
            (A::Sum(sum), _) => {
                let mut res = Sum::zero();
                for term in &sum.args {
                    let term_prod = Self::expand_binary_mul(term, rhs);
                    res.add_rhs(&term_prod)
                }
                A::Sum(res).into()
            }
            _ => Expr::prod([lhs, rhs]),
        }
    }

    fn distribute_first(&self) -> Expr {

        // prod = a * sum * b
        let sum_indx = if let Some(indx) = self.args.iter().position(|a| matches!(a.get(), Atom::Sum(_))) {
            indx
        } else {
            return Atom::Prod(self.clone()).into();
        };

        let mut a = Prod::one();
        let mut b = Prod::one();
        let sum = self.args[sum_indx].clone();

        for (i, arg) in self.args.iter().enumerate() {
            if i < sum_indx {
                a.args.push_back(arg.clone())
            } else if i > sum_indx {
                b.args.push_back(arg.clone())
            }
        }

        let (lhs, rhs): (Expr, Expr) = (Atom::Prod(a).into(), Atom::Prod(b).into());
        let mut res = Sum::zero();

        for term in sum.iter_args() {
            res.add_rhs(&(lhs.clone() * term * rhs.clone()));
        }

        Atom::Sum(res).into()
    }

    fn distribute(&self) -> Expr {
        self.args
            .iter()
            .fold(Atom::Prod(Prod::one()).into(), |l, r| {
                Self::expand_binary_mul(&l, r)
            })
    }

    fn reduce(&self) -> Expr {
        use Atom as A;
        if self.args.is_empty() {
            return Expr::one();
        } else if self.args.len() == 1 {
            return self.args.front().unwrap().clone();
        }

        let mut res = Prod::one();
        //res.args.push_back(Expr::one());

        let mut accum = Rational::ONE;

        for a in &self.args {
            match a.get() {
                A::Var(_) | A::Sum(_) | A::Prod(_) | A::Pow(_) => res.mul_rhs(a),
                A::Undef => return Expr::undef(),
                A::Rational(r) => accum *= r,
            }
        }

        if accum.is_zero() {
            return Expr::zero();
        }

        if !accum.is_one() {
            res.args.push_front(accum.into())
        }
        Atom::Prod(res).into()
    }
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub struct Pow {
    /// [base, exponent]
    pub(crate) args: [Expr; 2],
}

impl Pow {
    pub fn new(base: impl Borrow<Expr>, exponent: impl Borrow<Expr>) -> Pow {
        use Atom as A;
        let (base, exponent) = (base.borrow(), exponent.borrow());
        match (base.get(), exponent.get()) {
            (A::Undef, _) | (_, A::Undef) | (&A::ZERO, &A::ZERO) => Pow::undef(),
            _ => Pow {
                args: [base.clone(), exponent.clone()],
            },
        }
    }

    fn undef() -> Self {
        Pow {
            args: [Atom::Undef.into(), Atom::ONE.into()],
        }
    }

    pub fn base(&self) -> &Expr {
        &self.args[0]
    }

    pub fn exponent(&self) -> &Expr {
        &self.args[1]
    }

    fn expand(&self) -> Expr {
        use Atom as A;

        let (e, base) = match (self.exponent().get(), self.base().get()) {
            (A::Rational(r), A::Sum(sum))
                if r.is_int() && r.numer() >= &UInt::TWO && sum.args.len() > 1 =>
            {
                (r.numer().clone(), sum)
            }
            _ => return A::Pow(self.clone()).into(),
        };

        // (term + rest)^exp
        let exp = Expr::from(e.clone());
        let mut args = base.args.clone();
        let term = args.pop_front().unwrap();
        let rest = A::Sum(Sum { args }).into();

        let mut res = Sum::zero();
        for k in UInt::range_inclusive(UInt::ZERO, e.clone()) {
            if k == UInt::ZERO {
                // term^exp
                res.add_rhs(&Pow::new(&term, &exp).expand());
            } else if &k == &e {
                // rest^k
                res.add_rhs(&Pow::new(&rest, &Expr::from(k)).expand());
            } else {
                // term^k + rest^(exp-k)
                let c = UInt::binomial_coeff(&e, &k);
                let k_e = Expr::from(k.clone());
                let term_pow = Expr::from(c) * (&term).pow(k_e).expand();

                let rest_pow = (&rest).pow(&(e.clone() - &k).into()).expand();
                res.add_rhs(&Prod::expand_binary_mul(&term_pow, &rest_pow));
            }
        }

        Atom::Sum(res).into()
    }

    fn reduce(&self) -> Expr {
        use Atom as A;
        match (self.base().get(), self.exponent().get()) {
            (A::Undef, _) | (_, A::Undef) | (&A::ZERO, &A::ZERO) => Expr::undef(),
            (&A::ZERO, A::Rational(r)) if r.is_neg() => Expr::undef(),
            (A::Rational(b), A::Rational(e)) => {
                let (res, rem) = b.clone().pow(e.clone());
                if rem.is_zero() {
                    Expr::from(res)
                } else {
                    Expr::from(res) * Expr::from(b.clone()).pow(Expr::from(rem))
                }
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

macro_rules! bit {
    ($x:literal) => {
        1 << $x
    };
    ($x:ident) => {
        ExprFlags::$x.bits()
    };
}

bitflags::bitflags! {
    pub struct ExprFlags: u32 {
        const Zero          = bit!(1);
        const NonZero       = bit!(2);
        const Invertible    = bit!(3);

        const Scalar        = bit!(3);
        const Complex       = bit!(4);
        const Matrix        = bit!(5);
    }
}

impl ExprFlags {
    pub const fn is(&self, flag: ExprFlags) -> bool {
        self.contains(flag)
    }
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub struct Var(pub(crate) Rc<str>);
impl Var {
    fn len(&self) -> usize {
        self.0.len()
    }

    fn as_ref(&self) -> &str {
        self.0.as_ref()
    }
}
impl fmt::Debug for Var {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}
impl fmt::Display for Var {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum Real {
    Rational(Rational),
    Irrational(Irrational),
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum Irrational {
    E,
    PI,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum Func {
    Cos(Expr),
    Sin(Expr),
    /// [Func::Sin] / [Func::Cos]
    Tan(Expr),
    /// [Func::Cos] / [Func::Sin]
    Cot(Expr),
    /// 1 / [Func::Cos]
    Sec(Expr),
    /// 1 / [Func::Sin]
    Csc(Expr),

    Log {
        base: Irrational,
        arg: Expr,
    },

    Func(Var, Vec<Expr>),
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct Derivative {
    arg: Expr,
    var: Expr,
    degree: u64,
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub enum Atom {
    Undef,
    Rational(Rational),
    Var(Var),
    Sum(Sum),
    Prod(Prod),
    Pow(Pow),
    //Func(Func),
}

impl Atom {
    pub const ZERO: Atom = Atom::Rational(Rational::ZERO);
    pub const ONE: Atom = Atom::Rational(Rational::ONE);
}

impl fmt::Debug for Atom {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Atom::Undef => write!(f, "undef"),
            Atom::Rational(r) => write!(f, "{r}"),
            Atom::Var(v) => write!(f, "{v}"),
            Atom::Sum(sum) => write!(f, "{:?}", sum),
            Atom::Prod(prod) => write!(f, "{:?}", prod),
            Atom::Pow(pow) => write!(f, "{:?}", pow),
            //Atom::Func(func) => write!(f, "{:?}", func),
        }
    }
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub struct Expr(Rc<Atom>);

impl Expr {
    pub fn undef() -> Expr {
        Atom::Undef.into()
    }

    pub fn is_undef(&self) -> bool {
        matches!(self.get(), Atom::Undef)
    }

    pub fn zero() -> Expr {
        Atom::ZERO.into()
    }

    pub fn one() -> Expr {
        Atom::ONE.into()
    }

    pub fn rational<T: Into<Rational>>(r: T) -> Expr {
        Atom::Rational(r.into()).into()
    }

    pub fn add(lhs: impl Borrow<Expr>, rhs: impl Borrow<Expr>) -> Expr {
        use Atom as A;
        let (lhs, rhs): (&Expr, &Expr) = (lhs.borrow(), rhs.borrow());
        match (lhs.get(), rhs.get()) {
            (A::Undef, _) => A::Undef.into(),
            (_, A::Undef) => A::Undef.into(),
            (&A::ZERO, _) => rhs.clone(),
            (_, &A::ZERO) => lhs.clone(),
            (A::Rational(r1), A::Rational(r2)) => A::Rational(r1.clone() + r2).into(),
            (_, _) => Expr::sum([lhs, rhs]),
        }
        //Expr::sum([lhs.borrow(), rhs.borrow()])
    }
    pub fn sub(lhs: impl Borrow<Expr>, rhs: impl Borrow<Expr>) -> Expr {
        let (lhs, rhs) = (lhs.borrow(), rhs.borrow());
        let min_one = Expr::from(-1);
        let min_rhs = Expr::mul(min_one, rhs);
        Expr::add(lhs, min_rhs)
    }
    pub fn mul(lhs: impl Borrow<Expr>, rhs: impl Borrow<Expr>) -> Expr {
        use Atom as A;
        let (lhs, rhs): (&Expr, &Expr) = (lhs.borrow(), rhs.borrow());
        match (lhs.get(), rhs.get()) {
            (A::Undef, _) => A::Undef.into(),
            (_, A::Undef) => A::Undef.into(),
            (&A::ZERO, _) | (_, &A::ZERO) => Expr::zero(),
            (&A::ONE, _) => rhs.clone(),
            (_, &A::ONE) => lhs.clone(),
            (A::Rational(r1), A::Rational(r2)) => A::Rational(r1.clone() * r2).into(),
            (_, _) => Expr::prod([lhs, rhs]),
        }
        //Expr::prod([lhs.borrow(), rhs.borrow()])
    }
    pub fn div(lhs: impl Borrow<Expr>, rhs: impl Borrow<Expr>) -> Expr {
        let min_one = Expr::from(-1);
        let inv_rhs = Expr::pow(rhs, &min_one);
        Expr::mul(lhs, &inv_rhs)
    }

    pub fn sum<'a, T>(atoms: T) -> Expr
    where
        T: IntoIterator<Item = &'a Expr>,
    {
        let mut sum = Sum::zero();
        atoms.into_iter().for_each(|a| sum.add_rhs(a));
        Atom::Sum(sum).into()
    }

    pub fn prod<'a, T>(atoms: T) -> Expr
    where
        T: IntoIterator<Item = &'a Expr>,
    {
        let mut prod = Prod::one();
        atoms.into_iter().for_each(|a| prod.mul_rhs(a));
        Atom::Prod(prod).into()
    }

    pub fn pow(base: impl Borrow<Expr>, exponent: impl Borrow<Expr>) -> Expr {
        use Atom as A;
        let (base, exponent) = (base.borrow(), exponent.borrow());
        match (base.get(), exponent.get()) {
            (A::Undef, _) | (_, A::Undef) | (&A::ZERO, &A::ZERO) => Expr::undef(),
            (&A::ZERO, A::Rational(r)) if r.is_neg() => Expr::undef(),
            (_, &A::ONE) => base.clone(),
            (_, &A::ZERO) => Expr::one(),
            _ => Atom::Pow(Pow::new(base, exponent)).into(),
        }
    }

    pub fn free_of(&self, expr: &Expr) -> bool {
        self.iter_compl_sub_exprs().all(|e| e != expr)
    }

    pub fn free_of_set<'a, I: IntoIterator<Item = &'a Self>>(&'a self, exprs: I) -> bool {
        exprs.into_iter().all(|e| self.free_of(e))
    }

    pub fn substitude(&self, from: &Expr, to: &Expr) -> Self {
        self.concurr_substitude([(from, to)])
    }

    pub fn seq_substitude<'a, T>(&self, subs: T) -> Self
    where
        T: IntoIterator<Item = (&'a Expr, &'a Expr)>,
    {
        let mut res = self.clone();
        subs.into_iter().for_each(|(from, to)| {
            res.for_each_compl_sub_expr(|sub_expr| {
                if sub_expr == from {
                    *sub_expr = to.clone();
                }
            });
        });
        res
    }

    pub fn concurr_substitude<'a, T>(&self, subs: T) -> Self
    where
        T: IntoIterator<Item = (&'a Expr, &'a Expr)> + Copy,
    {
        let mut res = self.clone();
        res.try_for_each_compl_sub_expr(|sub_expr| {
            for (from, to) in subs {
                if sub_expr == from {
                    *sub_expr = (*to).clone();
                    return ops::ControlFlow::Break(());
                }
            }
            ops::ControlFlow::Continue(())
        });
        res
    }

    fn is_primitive(&self) -> bool {
        use Atom as A;
        match self.get() {
            A::Undef | A::Rational(_) => true,
            A::Var(_) | A::Sum(_) | A::Prod(_) | A::Pow(_) => false,
        }
    }

    // TODO: quot rule
    pub fn derivative<T: Borrow<Self>>(&self, x: T) -> Self {
        use Atom as A;
        let x = x.borrow();

        if self == x && !self.is_primitive() {
            return Expr::one();
        }

        match self.get() {
            A::Undef => self.clone(),
            A::Rational(_) => Expr::zero(),
            A::Sum(Sum { args }) => {
                let mut res = Sum::zero();
                args.iter()
                    .map(|a| a.derivative(x))
                    .for_each(|a| res.add_rhs(&a));
                Atom::Sum(res).into()
            }
            A::Prod(Prod { args }) => {
                if args.is_empty() {
                    return Expr::zero();
                } else if args.len() == 1 {
                    return args.front().unwrap().derivative(x);
                }
                // prod = term * args
                let mut args = args.clone();
                let term = args.pop_front().unwrap();
                let rest = Atom::Prod(Prod { args }).into();
                // d(a * b)/dx = da/dx * b + a * db/dx
                term.derivative(x) * &rest + term * rest.derivative(x)
            }
            A::Pow(pow) => {
                let v = pow.base();
                let w = pow.exponent();
                // d(v^w)/dx = w * v^(w - 1) * dv/dx + dw/dx * v^w * ln(v)
                w * v.pow(w - Expr::one()) * v.derivative(x) //TODO + w.derivative(x) * v.pow(w)
            }
            A::Var(_) => {
                if self.free_of(x) {
                    Expr::zero()
                } else {
                    todo!()
                }
            }
        }
    }

    pub fn variables_impl(&self, vars: &mut HashSet<Expr>) {
        use Atom as A;
        match self.get() {
            A::Rational(_) | A::Undef => (),
            A::Var(_) => {
                vars.insert(self.clone());
            }
            A::Sum(Sum { args }) => args.iter().for_each(|a| a.variables_impl(vars)),
            A::Prod(Prod { args }) => {
                args.iter().for_each(|a| {
                    if let A::Sum(_) = a.get() {
                        vars.insert(a.clone());
                    } else {
                        a.variables_impl(vars)
                    }
                });
            }
            A::Pow(pow) => {
                if let A::Rational(r) = pow.exponent().get() {
                    if r >= &Rational::ONE {
                        vars.insert(pow.base().clone());
                        return;
                    }
                }
                vars.insert(self.clone());
            }
        }
    }

    pub fn variables(&self) -> HashSet<Expr> {
        let mut vars = Default::default();
        self.variables_impl(&mut vars);
        vars
    }

    pub fn expand(&self) -> Self {
        use Atom as A;
        let mut expanded = self.clone();
        expanded.iter_args_mut().for_each(|e| *e = e.expand());

        match expanded.get() {
            A::Prod(prod) => prod.distribute(),
            A::Pow(pow) => pow.expand(),
            A::Undef | A::Rational(_) | A::Var(_) | A::Sum(_) => expanded,
        }
    }

    pub fn distribute(&self) -> Self {
        use Atom as A;
        if let A::Prod(prod) = self.get() {
            prod.distribute_first()
        } else {
            self.clone()
        }
        
    }

    pub fn reduce(&self) -> Self {
        use Atom as A;
        let mut res = self.clone();
        res.iter_args_mut().for_each(|a| *a = a.reduce());

        match res.get() {
            A::Undef | A::Rational(_) | A::Var(_) => res,
            A::Sum(sum) => sum.reduce(),
            A::Prod(prod) => prod.reduce(),
            A::Pow(pow) => pow.reduce(),
        }
    }

    //#[inline(always)]
    //pub fn as_monomial<'a>(&'a self, x: &'a Self) -> MonomialUV<'a> {
    //    MonomialUV { monom: self, var: x }
    //}
    #[inline(always)]
    pub fn as_monomial<'a>(&'a self, vars: &'a VarSet) -> MonomialView<'a> {
        MonomialView::new(self, vars)
    }
    #[inline(always)]
    pub fn as_polynomial<'a>(&'a self, vars: &'a VarSet) -> PolynomialView<'a> {
        PolynomialView::new(self, vars)
    }
    //#[inline(always)]
    //pub fn as_polynomial<'a>(&'a self, x: &'a Self) -> PolynomialUV<'a> {
    //    PolynomialUV { poly: self, var: x }
    //}

    pub fn fmt_ast(&self) -> fmt_ast::FmtAst<'_> {
        use Atom as A;
        match self.get() {
            A::Undef => FmtAst::Undef,
            A::Rational(r) => FmtAst::Rational(r.clone()),
            A::Var(v) => FmtAst::Var(v.as_ref()),
            A::Sum(sum) => sum
                .args
                .iter()
                .map(|a| a.fmt_ast())
                .reduce(|acc, e| acc + e)
                .unwrap_or(FmtAst::Sum(Default::default())),
            A::Prod(prod) => prod
                .args
                .iter()
                .map(|a| a.fmt_ast())
                .reduce(|acc, e| acc * e)
                .unwrap_or(FmtAst::Prod(Default::default())),
            A::Pow(pow) => pow.base().fmt_ast().pow(pow.exponent().fmt_ast()),
        }
    }

    pub fn fmt_unicode(&self) -> fmt_ast::UnicodeFmt {
        use fmt_ast::FormatWith;
        let mut fmt = fmt_ast::UnicodeFmt::default();
        self.fmt_ast().fmt_with(&mut fmt).unwrap();
        fmt
    }

    pub fn get(&self) -> &Atom {
        self.0.as_ref()
    }

    pub fn make_mut(&mut self) -> &mut Atom {
        Rc::make_mut(&mut self.0)
    }

    //fn args(&self) -> &[Self] {
    //    use Expr as E;
    //    match self.get() {
    //        A::Rational(_) | A::Var(_) | A::Undef => &[],
    //        A::Sum(Sum { args }) | A::Prod(Prod { args }) => args.as_slice(),
    //    }
    //}

    //fn args_mut(&mut self) -> &mut [Self] {
    //    use Expr as E;
    //    match self.make_mut() {
    //        A::Rational(_) | A::Var(_) | A::Undef => &mut [],
    //        A::Sum(Sum { args }) | A::Prod(Prod { args }) => args.as_mut_slice(),
    //    }
    //}

    fn iter_args(&self) -> Box<dyn Iterator<Item = &Self> + '_> {
        use Atom as A;
        match self.get() {
            A::Rational(_) | A::Var(_) | A::Undef => Box::new([].iter()),
            A::Sum(Sum { args }) | A::Prod(Prod { args }) => Box::new(args.iter()),
            A::Pow(Pow { args }) => Box::new(args.iter()),
        }
    }

    fn iter_args_mut(&mut self) -> Box<dyn Iterator<Item = &mut Self> + '_> {
        use Atom as A;
        match self.make_mut() {
            A::Rational(_) | A::Var(_) | A::Undef => Box::new([].iter_mut()),
            A::Sum(Sum { args }) | A::Prod(Prod { args }) => Box::new(args.iter_mut()),
            A::Pow(Pow { args }) => Box::new(args.iter_mut()),
        }
    }

    //fn n_args(&self) -> usize {
    //    self.iter_args().len()
    //}

    fn try_for_each_compl_sub_expr<F>(&mut self, func: F)
    where
        F: Fn(&mut Expr) -> ops::ControlFlow<()> + Copy,
    {
        if func(self).is_break() {
            return;
        }

        self.iter_args_mut()
            .for_each(|a| a.try_for_each_compl_sub_expr(func))
    }

    fn for_each_compl_sub_expr<F>(&mut self, func: F)
    where
        F: Fn(&mut Expr) + Copy,
    {
        self.try_for_each_compl_sub_expr(|expr| {
            func(expr);
            ops::ControlFlow::Continue(())
        });
    }

    fn iter_compl_sub_exprs(&self) -> ExprIterator<'_> {
        let atoms = vec![self];
        ExprIterator { atoms }
    }
}

impl<T: Borrow<Expr>> ops::Add<T> for &Expr {
    type Output = Expr;
    fn add(self, rhs: T) -> Self::Output {
        Expr::add(self, rhs)
    }
}
impl<T: Borrow<Expr>> ops::Add<T> for Expr {
    type Output = Expr;
    fn add(self, rhs: T) -> Self::Output {
        Expr::add(self, rhs)
    }
}
impl<T: Borrow<Expr>> ops::AddAssign<T> for Expr {
    fn add_assign(&mut self, rhs: T) {
        *self = &*self + rhs;
    }
}
impl<T: Borrow<Expr>> ops::Sub<T> for &Expr {
    type Output = Expr;
    fn sub(self, rhs: T) -> Self::Output {
        Expr::sub(self, rhs)
    }
}
impl<T: Borrow<Expr>> ops::Sub<T> for Expr {
    type Output = Expr;
    fn sub(self, rhs: T) -> Self::Output {
        Expr::sub(self, rhs)
    }
}
impl<T: Borrow<Expr>> ops::SubAssign<T> for Expr {
    fn sub_assign(&mut self, rhs: T) {
        *self = &*self - rhs;
    }
}
impl<T: Borrow<Expr>> ops::Mul<T> for &Expr {
    type Output = Expr;
    fn mul(self, rhs: T) -> Self::Output {
        Expr::mul(self, rhs)
    }
}
impl<T: Borrow<Expr>> ops::Mul<T> for Expr {
    type Output = Expr;
    fn mul(self, rhs: T) -> Self::Output {
        Expr::mul(self, rhs)
    }
}
impl<T: Borrow<Expr>> ops::MulAssign<T> for Expr {
    fn mul_assign(&mut self, rhs: T) {
        *self = &*self * rhs;
    }
}
impl<T: Borrow<Expr>> ops::Div<T> for &Expr {
    type Output = Expr;
    fn div(self, rhs: T) -> Self::Output {
        Expr::div(self, rhs)
    }
}
impl<T: Borrow<Expr>> ops::Div<T> for Expr {
    type Output = Expr;
    fn div(self, rhs: T) -> Self::Output {
        Expr::div(self, rhs)
    }
}
impl<T: Borrow<Expr>> ops::DivAssign<T> for Expr {
    fn div_assign(&mut self, rhs: T) {
        *self = &*self / rhs;
    }
}
impl<T: Borrow<Expr>> crate::utils::Pow<T> for &Expr {
    type Output = Expr;
    fn pow(self, rhs: T) -> Self::Output {
        Expr::pow(self, rhs)
    }
}
impl<T: Borrow<Expr>> crate::utils::Pow<T> for Expr {
    type Output = Expr;
    fn pow(self, rhs: T) -> Self::Output {
        Expr::pow(self, rhs)
    }
}

impl fmt::Debug for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self.0)
    }
}
impl fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        //let fmt_ast = self.fmt_ast();
        write!(f, "{}", self.fmt_unicode())
    }
}

impl From<&str> for Expr {
    fn from(value: &str) -> Self {
        Expr::from(Atom::Var(Var(Rc::from(value))))
    }
}
impl<T: Into<Rational>> From<T> for Expr {
    fn from(value: T) -> Self {
        Expr::from(Atom::Rational(value.into()))
    }
}

#[derive(Debug)]
struct ExprIterator<'a> {
    atoms: Vec<&'a Expr>,
}

impl<'a> Iterator for ExprIterator<'a> {
    type Item = &'a Expr;

    fn next(&mut self) -> Option<Self::Item> {
        self.atoms.pop().map(|expr| {
            expr.iter_args().for_each(|arg| self.atoms.push(arg));
            expr
        })
    }
}

#[cfg(test)]
mod test {
    use calcurs_macros::expr as e;

    use super::*;

    #[test]
    fn variables() {
        assert_eq!(
            e!(x ^ 3 + 3 * x ^ 2 * y + 3 * x * y ^ 2 + y ^ 3).variables(),
            [e!(x), e!(y)].into_iter().collect()
        );
        assert_eq!(
            e!(3 * x * (x + 1) * y ^ 2 * z ^ n).variables(),
            [e!(x), e!(x + 1), e!(y), e!(z ^ n)].into_iter().collect()
        );
        assert_eq!(
            e!(2 ^ (1 / 2) * x ^ 2 + 3 ^ (1 / 2) * x + 5 ^ (1 / 2)).variables(),
            [e!(x), e!(2 ^ (1 / 2)), e!(3 ^ (1 / 2)), e!(5 ^ (1 / 2))]
                .into_iter()
                .collect()
        );
    }

    #[test]
    fn reduce() {
        let checks = vec![
            (e!(1 + 2), e!(3)),
            (e!(a + undef), e!(undef)),
            (e!(a + (b + c)), e!(a + (b + c))),
            (e!(0 - 2 * b), e!((2 - 4) * b)),
        ];
        for (calc, res) in checks {
            assert_eq!(calc.reduce(), res);
        }
    }
}
