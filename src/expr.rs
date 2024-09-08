use std::{borrow::Borrow, collections::VecDeque, fmt, ops, rc::Rc};

use crate::{
    fmt_ast::{self, FmtAst},
    polynomial::{MonomialView, PolynomialView, VarSet},
    rational::{Int, Rational},
    utils::{self, HashSet},
};

use paste::paste;

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
            _ => self.args.push_back(rhs.clone()),
        }
    }

    fn reduce(&self) -> Expr {
        use Atom as A;
        if self.args.is_empty() {
            return Expr::zero();
        }

        let mut res = Sum::zero();
        let mut accum = Rational::ZERO;

        for a in &self.args {
            match a.get() {
                A::Irrational(_) | A::Func(_) | A::Var(_) | A::Sum(_) | A::Prod(_) | A::Pow(_) => {
                    res.add_rhs(a)
                }
                A::Undef => return Expr::undef(),
                A::Rational(r) => accum += r,
            }
        }

        //res.args[0] = accum.into();
        if !accum.is_zero() {
            res.args.push_front(accum.into());
        }
        if res.args.len() == 1 {
            return res.args.pop_front().unwrap();
        }

        Atom::Sum(res).into()
    }
}
impl fmt::Debug for Sum {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.args.is_empty() {
            return write!(f, "(+)");
        } else if self.args.len() == 1 {
            return write!(f, "(+{:?})", self.args.front().unwrap());
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
            A::Irrational(_) | A::Func(_) | A::Var(_) | A::Sum(_) | A::Pow(_) => {
                if let Some(arg) = self.args.iter_mut().find(|a| a.base() == rhs.base()) {
                    *arg = Expr::pow(arg.base(), arg.exponent() + rhs.exponent())
                } else {
                    self.args.push_back(rhs.clone())
                }
            }
        }
    }

    fn expand_mul(lhs: &Expr, rhs: &Expr) -> Expr {
        use Atom as A;
        match (lhs.get(), rhs.get()) {
            (A::Prod(_), A::Prod(_)) => {
                let mut res = Prod::one();
                res.mul_rhs(lhs);
                res.mul_rhs(rhs);
                A::Prod(res).into()
            }
            (A::Sum(sum), _) => {
                let mut res = Sum::zero();
                for term in &sum.args {
                    let term_prod = Self::expand_mul(term, rhs);
                    res.add_rhs(&term_prod)
                }
                A::Sum(res).into()
            }
            (_, A::Sum(sum)) => {
                let mut res = Sum::zero();
                for term in &sum.args {
                    let term_prod = Self::expand_mul(lhs, term);
                    res.add_rhs(&term_prod)
                }
                A::Sum(res).into()
            }
            _ => Expr::mul(lhs, rhs),
        }
    }

    fn distribute_first(&self) -> Expr {
        // prod = a * sum * b
        let sum_indx = if let Some(indx) = self
            .args
            .iter()
            .position(|a| matches!(a.get(), Atom::Sum(_)))
        {
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
                Self::expand_mul(&l, r)
            })
    }

    fn reduce(&self) -> Expr {
        use Atom as A;
        if self.args.is_empty() {
            return Expr::one();
        }
        let mut res = Prod::one();
        let mut accum = Rational::ONE;

        for a in &self.args {
            match a.get() {
                A::Func(_) | A::Irrational(_) | A::Var(_) | A::Sum(_) | A::Prod(_) | A::Pow(_) => res.mul_rhs(a),
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
        if res.args.len() == 1 {
            return res.args.pop_front().unwrap();
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
    pub fn base(&self) -> &Expr {
        &self.args[0]
    }

    pub fn exponent(&self) -> &Expr {
        &self.args[1]
    }

    pub fn expand_pow_rec(&self, recurse: bool) -> Expr {
        use Atom as A;
        let expand_pow = |lhs: &Expr, rhs: &Expr| -> Expr {
            if recurse {
                Expr::pow(lhs, rhs).expand()
            } else {
                Expr::pow(lhs, rhs)
            }
        };
        let expand_mul = |lhs: &Expr, rhs: &Expr| -> Expr {
            if recurse {
                Expr::mul_expand(lhs, rhs)
            } else {
                Expr::mul(lhs, rhs)
            }
        };

        let (e, base) = match (self.base().get(), self.exponent().get()) {
            (A::Sum(sum), A::Rational(r))
                if r.is_int() && r > &Rational::ONE && sum.args.len() > 1 =>
            {
                (r.numer().clone(), sum)
            }
            (A::Sum(sum), A::Rational(r)) if r > &Rational::ONE && sum.args.len() > 1 => {
                let (div, rem) = r.div_rem();
                return expand_mul(
                    &expand_pow(self.base(), &Expr::from(div)),
                    &expand_pow(self.base(), &Expr::from(rem)),
                );
            }
            (A::Prod(Prod { args }), _) => {
                return args
                    .iter()
                    .map(|a| {
                        if recurse {
                            Expr::pow(a, self.exponent()).expand()
                        } else {
                            Expr::pow(a, self.exponent())
                        }
                    })
                    .fold(Expr::one(), |prod, rhs| prod * rhs)
            }
            _ => {
                return A::Pow(self.clone()).into();
            }
        };

        // (a + b)^exp
        let exp = Expr::from(e.clone());
        let mut args = base.args.clone();
        let a = args.pop_front().unwrap();
        let b = A::Sum(Sum { args }).into();

        let mut res = Sum::zero();
        for k in Int::range_inclusive(Int::ZERO, e.clone()) {
            let rhs = if k == Int::ZERO {
                // 1 * a^exp
                expand_pow(&a, &exp)
            } else if &k == &e {
                // 1 * b^k
                expand_pow(&b, &Expr::from(k.clone()))
            } else {
                // a^k + b^(exp-k)
                let c = Int::binomial_coeff(&e, &k);
                let k_e = Expr::from(k.clone());

                expand_mul(
                    &Expr::from(c),
                    &expand_mul(
                        &expand_pow(&a, &k_e),
                        &expand_pow(&b, &Expr::from(e.clone() - &k)),
                    ),
                )
            };
            res.add_rhs(&rhs);
        }

        Atom::Sum(res).into()
    }

    pub fn expand_pow(&self) -> Expr {
        self.expand_pow_rec(true)
    }

    fn reduce(&self) -> Expr {
        use Atom as A;
        match (self.base().get(), self.exponent().get()) {
            (A::Undef, _) | (_, A::Undef) | (&A::ZERO, &A::ZERO) => Expr::undef(),
            (&A::ZERO, A::Rational(r)) if r.is_neg() => Expr::undef(),
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
pub struct Var(pub(crate) PTR<str>);
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
pub enum Real {
    Rational(Rational),
    Irrational(Irrational),
}

impl fmt::Display for Real {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Real::Rational(r) => write!(f, "{r}"),
            Real::Irrational(i) => write!(f, "{i}"),
        }
    }
}

impl From<Rational> for Real {
    fn from(value: Rational) -> Self {
        Real::Rational(value)
    }
}
impl From<Irrational> for Real {
    fn from(value: Irrational) -> Self {
        Real::Irrational(value)
    }
}
impl From<Real> for Expr {
    fn from(value: Real) -> Self {
        match value {
            Real::Rational(r) => r.into(),
            Real::Irrational(i) => i.into(),
        }
    }
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub enum Irrational {
    E,
    PI,
}

impl From<Irrational> for Expr {
    fn from(value: Irrational) -> Self {
        Expr::from(Atom::Irrational(value))
    }
}

impl fmt::Debug for Irrational {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self}")
    }
}
impl fmt::Display for Irrational {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let sym = match self {
            Irrational::E => "e",
            Irrational::PI => "pi",
        };
        write!(f, "{}", sym)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Func {
    Sin(Expr),
    Cos(Expr),
    Tan(Expr),
    Sec(Expr),
    ArcSin(Expr),
    ArcCos(Expr),
    ArcTan(Expr),
    Exp(Expr),
    Log(Real, Expr),
}

macro_rules! func_fn {
    ($name: ident) => {
        pub fn $name(e: impl Borrow<Expr>) -> Expr {
            paste! {
            Expr::from(Atom::Func(Func::[<$name:camel>](e.borrow().clone())))
            }
        }
    };
}

impl Func {
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
            Func::Log(base, _) => format!("{base}"),
        }
    }
    pub fn derivative(&self, x: impl Borrow<Expr>) -> Expr {
        use Expr as E;
        use Func as F;
        let d = |e: &Expr| -> Expr { e.derivative(x) };
        match self {
            F::Sin(f) => d(f) * E::cos(f),
            F::Cos(f) => d(f) * E::from(-1) * E::sin(f),
            F::Tan(f) => d(f) * E::pow(E::sec(f), &2.into()),
            F::Sec(f) => d(f) * E::tan(f) * E::sec(f),
            F::ArcSin(f) => {
                d(f) * E::pow(E::from(1) - E::pow(f, E::from(2)), E::from((-1i64, 2i64)))
            }
            F::ArcCos(f) => {
                d(f) * E::from(-1)
                    * E::pow(E::from(1) - E::pow(f, E::from(2)), E::from((-1i64, 2i64)))
            }
            F::ArcTan(f) => {
                d(f) * E::pow(E::from(1) + E::pow(f, E::from(2)), E::from(-1))
            }
            F::Log(base, f) => d(f) * Expr::pow(f * E::ln(Expr::from(base.clone())), E::from(-1)),
            F::Exp(f) => Expr::exp(f) * d(f),
        }
    }

    pub fn iter_args(&self) -> Box<dyn Iterator<Item = &Expr> + '_> {
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
            | F::Log(_, x) => Box::new(std::iter::once(x)),
        }
    }
    pub fn iter_args_mut(&mut self) -> Box<dyn Iterator<Item = &mut Expr> + '_> {
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
            | F::Log(_, x) => Box::new(std::iter::once(x)),
        }
    }

    pub fn reduce(&self) -> Expr {
        use Func as F;
        match self {
            F::Log(base, x) if x.check_real(base) => Expr::one(),
            _ => Atom::Func(self.clone()).into(),
        }
    }
}

//#[derive(Debug, Clone, PartialEq, Eq, Hash)]
//pub struct Func {
//    kind: FuncKind,
//    args: Vec<Expr>,
//}

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
    Irrational(Irrational),
    Var(Var),
    Sum(Sum),
    Prod(Prod),
    Pow(Pow),
    Func(Func),
}

impl Atom {
    pub const MINUS_ONE: Atom = Atom::Rational(Rational::MINUS_ONE);
    pub const ZERO: Atom = Atom::Rational(Rational::ZERO);
    pub const ONE: Atom = Atom::Rational(Rational::ONE);
}

impl fmt::Debug for Atom {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Atom::Undef => write!(f, "undef"),
            Atom::Rational(r) => write!(f, "{r}"),
            Atom::Irrational(r) => write!(f, "{r}"),
            Atom::Var(v) => write!(f, "{v}"),
            Atom::Sum(sum) => write!(f, "{:?}", sum),
            Atom::Prod(prod) => write!(f, "{:?}", prod),
            Atom::Pow(pow) => write!(f, "{:?}", pow),
            Atom::Func(func) => write!(f, "{func:?}"),
            //Atom::Func(func) => write!(f, "{:?}", func),
        }
    }
}

pub(crate) type PTR<T> = Rc<T>;

#[derive(Clone, PartialEq, Eq, Hash)]
pub struct Expr(PTR<Atom>);

//static E_ONE: LazyLock<Expr> = LazyLock::new(|| {
//    Expr::from(Atom::ONE)
//});
//static E_ZERO: LazyLock<Expr> = LazyLock::new(|| {
//    Expr::from(Atom::ZERO)
//});

impl Expr {
    pub fn undef() -> Expr {
        Atom::Undef.into()
    }

    pub fn is_undef(&self) -> bool {
        matches!(self.get(), Atom::Undef)
    }

    pub fn check_real(&self, r: &Real) -> bool {
        match (self.get(), r) {
            (Atom::Rational(r1), Real::Rational(r2)) => r1 == r2,
            (Atom::Irrational(i1), Real::Irrational(i2)) => i1 == i2,
            _ => false,
        }
    }

    pub fn zero() -> Expr {
        Atom::ZERO.into()
    }
    //pub fn zero_ref() -> &'static Expr {
    //    &E_ZERO
    //}

    pub fn one() -> Expr {
        Atom::ONE.into()
    }

    pub fn var(str: &str) -> Expr {
        Expr::from(str)
    }
    //pub fn one_ref() -> &'static Expr {
    //    &E_ONE
    //}
    pub fn min_one() -> Expr {
        Atom::MINUS_ONE.into()
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
            (_, _) => {
                let mut sum = Sum::zero();
                sum.add_rhs(lhs);
                sum.add_rhs(rhs);
                A::Sum(sum).into()
            }
        }
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
            (A::Undef, _) | (_, A::Undef) => A::Undef.into(),
            (&A::ZERO, _) | (_, &A::ZERO) => Expr::zero(),
            (&A::ONE, _) => rhs.clone(),
            (_, &A::ONE) => lhs.clone(),
            (A::Rational(r1), A::Rational(r2)) => A::Rational(r1.clone() * r2).into(),
            (_, _) => {
                if lhs.base() == rhs.base() {
                    return Expr::pow(lhs.base(), lhs.exponent() + rhs.exponent());
                } else {
                    let mut prod = Prod::one();
                    prod.mul_rhs(lhs);
                    prod.mul_rhs(rhs);
                    A::Prod(prod).into()
                }
                //Expr::prod([lhs, rhs]),
            }
        }
        //Expr::prod([lhs.borrow(), rhs.borrow()])
    }
    pub fn div(lhs: impl Borrow<Expr>, rhs: impl Borrow<Expr>) -> Expr {
        let min_one = Expr::from(-1);
        let inv_rhs = Expr::pow(rhs, &min_one);
        Expr::mul(lhs, &inv_rhs)
    }

    func_fn!(cos);
    func_fn!(sin);
    func_fn!(tan);
    func_fn!(sec);
    func_fn!(arc_sin);
    func_fn!(arc_cos);
    func_fn!(arc_tan);
    func_fn!(exp);

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

    //fn sum<'a, T>(atoms: T) -> Expr
    //where
    //    T: IntoIterator<Item = &'a Expr>,
    //{
    //    let mut sum = Sum::zero();
    //    atoms.into_iter().for_each(|a| sum.add_rhs(a));
    //    Atom::Sum(sum).into()
    //}

    //fn prod<'a, T>(atoms: T) -> Expr
    //where
    //    T: IntoIterator<Item = &'a Expr>,
    //{
    //    let mut prod = Prod::one();
    //    atoms.into_iter().for_each(|a| prod.mul_rhs(a));
    //    Atom::Prod(prod).into()
    //}

    pub fn raw_pow(base: impl Borrow<Expr>, exponent: impl Borrow<Expr>) -> Expr {
        Atom::Pow(Pow {
            args: [base.borrow().clone(), exponent.borrow().clone()],
        })
        .into()
    }

    pub fn pow(base: impl Borrow<Expr>, exponent: impl Borrow<Expr>) -> Expr {
        use Atom as A;
        let (base, exponent) = (base.borrow(), exponent.borrow());
        match (base.get(), exponent.get()) {
            (A::Undef, _) | (_, A::Undef) | (&A::ZERO, &A::ZERO) => Expr::undef(),
            (&A::ZERO, A::Rational(r)) if r.is_neg() => Expr::undef(),
            (&A::ONE, _) => Expr::one(),
            (_, &A::ONE) => base.clone(),
            (_, &A::ZERO) => Expr::one(),
            (A::Rational(b), A::Rational(e)) if b.is_int() && e.is_int() => {
                let (pow, rem) = b.clone().pow(e.clone());
                assert!(rem.is_zero());
                Expr::from(pow)
            }
            (A::Pow(pow), A::Rational(e)) if e.is_int() => {
                Expr::pow(pow.base(), pow.exponent() * exponent)
            }
            //(A::Pow(pow), A::Rational(e2)) if e2.is_int() => {
            //    println!("pow: {base:?}, {exponent:?}");
            //    match pow.exponent().get() {
            //        A::Rational(e1) => {
            //            Expr::pow(base, Expr::from(e1.clone() * e2))
            //        }
            //        _ => Expr::raw_pow(base, exponent),
            //    }
            //}
            _ => Expr::raw_pow(base, exponent),
        }
    }
    pub fn pow_expand(base: impl Borrow<Expr>, exponent: impl Borrow<Expr>) -> Expr {
        Expr::pow(base, exponent).expand()
    }

    pub fn sqrt(v: impl Borrow<Expr>) -> Expr {
        Expr::pow(v, Expr::from((1u64, 2u64)))
    }

    pub fn mul_expand(lhs: impl Borrow<Expr>, rhs: impl Borrow<Expr>) -> Expr {
        use Atom as A;
        let (lhs, rhs) = (lhs.borrow(), rhs.borrow());
        match (lhs.get(), rhs.get()) {
            (A::Prod(_), A::Prod(_)) => {
                let mut res = Prod::one();
                res.mul_rhs(lhs);
                res.mul_rhs(rhs);
                A::Prod(res).into()
            }
            (A::Sum(sum), _) => {
                let mut res = Sum::zero();
                for term in &sum.args {
                    let term_prod = Self::mul_expand(term, rhs);
                    res.add_rhs(&term_prod)
                }
                A::Sum(res).into()
            }
            (_, A::Sum(sum)) => {
                let mut res = Sum::zero();
                for term in &sum.args {
                    let term_prod = Self::mul_expand(lhs, term);
                    res.add_rhs(&term_prod)
                }
                A::Sum(res).into()
            }
            _ => Expr::mul(lhs, rhs),
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
            A::Irrational(_) | A::Undef | A::Rational(_) => true,
            A::Func(_) | A::Var(_) | A::Sum(_) | A::Prod(_) | A::Pow(_) => false,
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
            A::Irrational(_) | A::Rational(_) => Expr::zero(),
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
                w * Expr::pow(v, w - Expr::one()) * v.derivative(x)
                    + w.derivative(x) * Expr::pow(v, w) * Expr::ln(v)
            }
            A::Var(_) => {
                if self.free_of(x) {
                    Expr::zero()
                } else {
                    todo!()
                }
            }
            A::Func(f) => f.derivative(x),
        }
    }

    pub fn variables_impl(&self, vars: &mut HashSet<Expr>) {
        use Atom as A;
        match self.get() {
            A::Irrational(_) | A::Rational(_) | A::Undef => (),
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
            A::Func(_) => todo!(),
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
        expanded.iter_args_mut().for_each(|a| *a = a.expand());
        match expanded.get() {
            A::Var(_) | A::Undef | A::Rational(_) => expanded.clone(),

            A::Sum(Sum { args }) if args.len() == 1 => args.front().unwrap().expand(),
            A::Prod(Prod { args }) if args.len() == 1 => args.front().unwrap().expand(),
            //A::Sum(Sum { args }) => A::Sum(Sum {
            //    args: args.iter().map(|e| e.expand()).collect(),
            //})
            //.into(),
            A::Prod(prod) => prod.distribute(),
            //args
            //.iter()
            //.map(|e| e.expand())
            //.fold(Expr::one(), |prod, rhs| Prod::expand_mul(&prod, &rhs)),
            A::Pow(pow) => match pow.exponent().get() {
                A::Rational(r) if /*r.is_int() &&*/ r > &Rational::ONE => pow.expand_pow(),
                A::Rational(r) if r == &Rational::ONE => return pow.base().clone(),
                _ => expanded.clone(),
            },
            _ => expanded.clone(),
        }
        //let mut expanded = self.clone();
        //expanded.iter_args_mut().for_each(|e| *e = e.expand());

        //match expanded.get() {
        //    A::Prod(prod) => prod.distribute(),
        //    A::Pow(pow) => pow.expand(),
        //    A::Undef | A::Rational(_) | A::Var(_) | A::Sum(_) => expanded,
        //}
    }

    pub fn expand_main_op(&self) -> Self {
        use Atom as A;
        match self.get() {
            A::Prod(prod) => prod.distribute(),
            A::Pow(pow) => pow.expand_pow_rec(false),
            A::Irrational(_) | A::Undef | A::Rational(_) | A::Var(_) | A::Sum(_) => self.clone(),
            A::Func(_) => todo!(),
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
            A::Irrational(_) | A::Undef | A::Rational(_) | A::Var(_) => res,
            A::Sum(sum) => sum.reduce(),
            A::Prod(prod) => prod.reduce(),
            A::Pow(pow) => pow.reduce(),
            A::Func(func) => func.reduce(),
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

    pub fn numerator(&self) -> Expr {
        use Atom as A;
        match self.get() {
            A::Undef => self.clone(),
            A::Rational(r) => r.numer().into(),
            A::Pow(pow) => match pow.exponent().get() {
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
        match self.get() {
            A::Undef => self.clone(),
            A::Rational(r) => r.denom().into(),
            A::Pow(pow) => match pow.exponent().get() {
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

    fn rationalize_add(lhs: &Self, rhs: &Self) -> Expr {
        let ln = lhs.numerator();
        let ld = lhs.denominator();
        let rn = rhs.numerator();
        let rd = rhs.denominator();
        if ld.get() == &Atom::ONE && rd.get() == &Atom::ONE {
            lhs + rhs
        } else {
            Self::rationalize_add(&(ln * &rd), &(rn * &ld)) / (ld * rd)
        }
    }

    pub fn rationalize(&self) -> Expr {
        use Atom as A;
        match self.get() {
            A::Prod(_) => {
                let mut r = self.clone();
                r.iter_args_mut().for_each(|a| *a = a.rationalize());
                r
            }
            A::Sum(Sum { args }) => args
                .iter()
                .map(|a| a.rationalize())
                .fold(Expr::zero(), |sum, r| Self::rationalize_add(&sum, &r)),
            A::Pow(pow) => Expr::pow(pow.base().rationalize(), pow.exponent()),
            _ => self.clone(),
        }
    }

    /// divide lhs and rhs by their common factor and
    /// return them in the form (fac, (lhs/fac, rhs/fac)
    pub fn factorize_common_terms(lhs: &Expr, rhs: &Self) -> (Expr, (Expr, Expr)) {
        use Atom as A;
        if lhs == rhs {
            return (lhs.clone(), (Expr::one(), Expr::one()));
        }
        match (lhs.get(), rhs.get()) {
            (A::Rational(r1), A::Rational(r2)) if r1.is_int() && r2.is_int() => {
                let (i1, i2) = (r1.to_int().unwrap(), r2.to_int().unwrap());
                let gcd = i1.gcd(&i2);
                let rgcd = Rational::from(gcd);
                let l = (r1.clone() / &rgcd).unwrap();
                let r = (r2.clone() / &rgcd).unwrap();
                (rgcd.into(), (l.into(), r.into()))
            }
            (A::Prod(Prod { args }), _) if !args.is_empty() => {
                if args.len() == 1 {
                    let lhs = args.front().unwrap();
                    return Self::factorize_common_terms(lhs, rhs);
                }
                /*
                (a*x) * (b*y), (u*x*y)
                => common(a*x, u*x*y) -> (x, (a, u*y))
                => common(b*y, u*y) -> (y, (b, u))
                => return (x*y, (a*b, u))

                */
                let mut args = args.clone();
                let uxy = rhs;
                let ax = args.pop_front().unwrap();
                let by = if args.len() == 1 {
                    args.pop_front().unwrap()
                } else {
                    Expr::from(A::Prod(Prod { args }))
                };
                let (x, (a, uy)) = Self::factorize_common_terms(&ax, uxy);
                let (y, (b, u)) = Self::factorize_common_terms(&by, &uy);
                (x * y, (a * b, u))
            }
            (A::Sum(Sum { args }), _) if !args.is_empty() => {
                if args.len() == 1 {
                    let lhs = args.front().unwrap();
                    return Self::factorize_common_terms(lhs, rhs);
                }
                /*
                abxy + cdxy, u*x*y*a*c
                => common(abxy, uacxy) -> (axy, (b, uc))
                => common(cdxy, uacxy) -> (cxy, (d, ua))
                => common(axy, cxy)    -> (xy, (a, c))
                => common(ua, uc)      -> (u, (a, c))
                => return (xy, (ab + cd, uac))
                */
                let mut args = args.clone();
                let uacxy = rhs;
                let abxy = args.pop_front().unwrap();
                let cdxy = if args.len() == 1 {
                    args.pop_front().unwrap()
                } else {
                    Expr::from(A::Sum(Sum { args }))
                };

                let (axy, (b, uc)) = Self::factorize_common_terms(&abxy, uacxy);
                let (cxy, (d, ua)) = Self::factorize_common_terms(&cdxy, uacxy);
                let (xy, (a, c)) = Self::factorize_common_terms(&axy, &cxy);
                let (u, (_a, _c)) = Self::factorize_common_terms(&ua, &uc);
                /*
                println!("abxy + cdxy       : {lhs:?}");
                println!("uacxy             : {uacxy:?}");
                println!("abxy              : {abxy:?}");
                println!("cdxy              : {cdxy:?}");
                println!("(axy, (b, uc))    : ({axy:?}, ({b:?}, {uc:?}))");
                println!("(cxy, (d, ua))    : ({cxy:?}, ({d:?}, {ua:?}))");
                println!("(xy, (a, c))      : ({xy:?}, ({a:?}, {c:?}))");
                println!("(u, (_a, _c))     : ({u:?}, ({_a:?}, {_c:?}))");
                println!("");
                */
                (xy, (a * b + c * d, u * _a * _c))
            }
            (_, A::Sum(_) | A::Prod(_)) => {
                let (fac, (r, l)) = Self::factorize_common_terms(rhs, lhs);
                (fac, (l, r))
            }
            (_, _) => match (lhs.exponent().get(), rhs.exponent().get()) {
                (A::Rational(r1), A::Rational(r2))
                    if r1.is_pos() && r2.is_pos() && rhs.base() == lhs.base() =>
                {
                    let e = std::cmp::min(r1, r2).clone();
                    let b = rhs.base();
                    (
                        Expr::pow(&b, Expr::from(e.clone())),
                        (
                            Expr::pow(&b, Expr::from(r1.clone() - &e)),
                            Expr::pow(&b, Expr::from(r2.clone() - e)),
                        ),
                    )
                }
                _ => (Expr::one(), (lhs.clone(), rhs.clone())),
            },
        }
    }

    pub fn common_factors(lhs: &Self, rhs: &Self) -> Expr {
        Expr::factorize_common_terms(lhs, rhs).0
    }

    pub fn factor_out(&self) -> Expr {
        use Atom as A;
        match self.get() {
            A::Prod(Prod { args }) => args
                .iter()
                .map(|a| a.factor_out())
                .fold(Expr::one(), |prod, rhs| prod * rhs),
            A::Pow(pow) => Expr::pow(pow.base().factor_out(), pow.exponent()),
            A::Sum(Sum { args }) => {
                let s = args
                    .iter()
                    .map(|a| a.factor_out())
                    .fold(Expr::zero(), |sum, rhs| sum + rhs)
                    .reduce();
                if let A::Sum(Sum { args }) = s.get() {
                    // sum = a + b
                    let mut args = args.clone();
                    let a = args.pop_front().unwrap();
                    let b = if args.len() == 1 {
                        args.pop_front().unwrap()
                    } else {
                        Expr::from(A::Sum(Sum { args }))
                    };
                    let (f, (a_div_f, b_div_f)) = Expr::factorize_common_terms(&a, &b);
                    f * (a_div_f + b_div_f)
                } else {
                    s
                }
            }
            _ => self.clone(),
        }
    }

    pub fn cancel(&self) -> Expr {
        let n = self.numerator();
        let d = self.denominator();
        n.factor_out() / d.factor_out()
    }

    pub fn fmt_ast(&self) -> FmtAst {
        use Atom as A;
        match self.get() {
            A::Undef => FmtAst::Undef,
            A::Rational(r) => FmtAst::Rational(r.clone()),
            A::Var(v) => FmtAst::Var(v.0.to_string()),
            A::Irrational(ir) => FmtAst::Var(ir.to_string()),
            A::Sum(sum) => {
                let iter: Box<dyn Iterator<Item = &Expr>> =
                    if let Some(Atom::Rational(_)) = sum.first() {
                        // put rational last
                        Box::new(sum.args.iter().skip(1).chain(sum.args.iter().take(1)))
                    } else {
                        Box::new(sum.args.iter())
                    };
                iter.map(|a| a.fmt_ast())
                    .reduce(|acc, e| acc + e)
                    .unwrap_or(FmtAst::Prod(Default::default()))
            }
            A::Prod(prod) => prod
                .args
                .iter()
                .map(|a| a.fmt_ast())
                .reduce(|acc, e| acc * e)
                .unwrap_or(FmtAst::Prod(Default::default())),
            A::Pow(pow) => pow.base().fmt_ast().pow(pow.exponent().fmt_ast()),
            A::Func(func) => FmtAst::Func(fmt_ast::Func(
                func.name(),
                func.iter_args().map(|a| a.fmt_ast()).collect(),
            )),
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

    pub fn get_flatten(&self) -> &Atom {
        match self.0.as_ref() {
            Atom::Sum(Sum { args }) | Atom::Prod(Prod { args }) if args.len() == 1 => {
                args.front().unwrap().get()
            }
            a => a,
        }
    }

    pub fn make_mut(&mut self) -> &mut Atom {
        PTR::make_mut(&mut self.0)
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
            A::Irrational(_) | A::Rational(_) | A::Var(_) | A::Undef => Box::new([].iter()),
            A::Sum(Sum { args }) | A::Prod(Prod { args }) => Box::new(args.iter()),
            A::Pow(Pow { args }) => Box::new(args.iter()),
            A::Func(func) => func.iter_args(),
        }
    }

    fn iter_args_mut(&mut self) -> Box<dyn Iterator<Item = &mut Self> + '_> {
        use Atom as A;
        match self.make_mut() {
            A::Irrational(_) | A::Rational(_) | A::Var(_) | A::Undef => Box::new([].iter_mut()),
            A::Sum(Sum { args }) | A::Prod(Prod { args }) => Box::new(args.iter_mut()),
            A::Pow(Pow { args }) => Box::new(args.iter_mut()),
            A::Func(func) => func.iter_args_mut(),
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
//impl<T: Borrow<Expr>> crate::utils::Pow<T> for &Expr {
//    type Output = Expr;
//    fn pow(self, rhs: T) -> Self::Output {
//        Expr::pow(self, rhs)
//    }
//}
//impl<T: Borrow<Expr>> crate::utils::Pow<T> for Expr {
//    type Output = Expr;
//    fn pow(self, rhs: T) -> Self::Output {
//        Expr::pow(self, rhs)
//    }
//}

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
        Expr::from(Atom::Var(Var(PTR::from(value))))
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
    use assert_eq as eq;
    use calcurs_macros::expr as e;

    use super::*;

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
    fn expand() {
        eq!(
            e!(x * (2 + (1 + x) ^ 2)).expand_main_op(),
            e!(2 * x + x * (1 + x) ^ 2)
        );
        eq!(
            e!((x + (1 + x) ^ 2) ^ 2).expand_main_op().reduce(),
            e!(x ^ 2 + 2 * x * (1 + x) ^ 2 + (1 + x) ^ 4)
        );
        eq!(
            e!((x + 2) * (x + 3) * (x + 4))
                .expand()
                .as_polynomial(&[e!(x)].into())
                .collect_terms(),
            Some(e!(x ^ 3 + 9 * x ^ 2 + 26 * x + 24))
        );
        eq!(
            e!((x + 1) ^ 2 + (y + 1) ^ 2).expand().reduce(),
            e!(2 + (2 * x) + x ^ 2 + (2 * y) + y ^ 2)
        );
        //eq!(
        //    e!(((x + 2) ^ 2 + 3) ^ 2).expand().reduce(),
        //    e!(x ^ 4 + 8 ^ 3 + 30 * x ^ 2 + 56 * x + 49)
        //);
    }

    #[test]
    fn reduce() {
        let checks = vec![
            (e!(1 + 2), e!(3)),
            (e!(a + undef), e!(undef)),
            (e!(a + (b + c)), e!(a + (b + c))),
            (e!(0 - 2 * b), e!((2 - 4) * b)),
            (e!(a + 0), e!(a)),
            (e!(0 + a), e!(a)),
            (e!(1 + 2), e!(3)),
            (e!(x + 0), e!(x)),
            (e!(0 + x), e!(x)),
            (e!(0 - x), e!((4 - 5) * x)),
            (e!(x - 0), e!(x)),
            (e!(3 - 2), e!(1)),
            (e!(x * 0), e!(0)),
            (e!(0 * x), e!(0)),
            (e!(x * 1), e!(x)),
            (e!(1 * x), e!(x)),
            (e!(0 ^ 0), e!(undef)),
            (e!(0 ^ 1), e!(0)),
            (e!(0 ^ 314), e!(0)),
            (e!(1 ^ 0), e!(1)),
            (e!(314 ^ 0), e!(1)),
            (e!(314 ^ 1), e!(314)),
            (e!(x ^ 1), e!(x)),
            (e!(1 ^ x), e!(1)),
            (e!(1 ^ 314), e!(1)),
            (e!(3 ^ 3), e!(27)),
            (e!(a - b), e!(a + ((2 - 3) * b))),
            (e!(a / b), e!(a * b ^ (2 - 3))),
        ];
        for (calc, res) in checks {
            eq!(calc.reduce(), res);
        }
    }

    #[test]
    fn distributive() {
        eq!(
            e!(a * (b + c) * (d + e)).distribute(),
            e!(a * b * (d + e) + a * c * (d + e))
        );
        eq!(
            e!((x + y) / (x * y)).distribute(),
            e!(x / (x * y) + y / (x * y))
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
    fn rationalize() {
        eq!(e!((1 + 1 / x) ^ 2).rationalize(), e!(((x + 1) / x) ^ 2));
        eq!(
            e!((1 + 1 / x) ^ (1 / 2)).rationalize(),
            e!(((x + 1) / x) ^ (1 / 2))
        );
    }

    #[test]
    fn common_factors() {
        eq!(
            Expr::factorize_common_terms(&e!(6 * x * y ^ 3), &e!(2 * x ^ 2 * y * z)),
            (e!(2 * x * y), (e!(3 * y ^ 2), e!(x * z)))
        );
        eq!(
            Expr::factorize_common_terms(&e!(a * (x + y)), &e!(x + y)),
            (e!(x + y), (e!(a), e!(1)))
        );
    }

    #[test]
    fn factor_out() {
        eq!(
            e!((x ^ 2 + x * y) ^ 3).factor_out().expand_main_op(),
            e!(x ^ 3 * (x + y) ^ 3)
        );
        eq!(e!(a * (b + b * x)).factor_out(), e!(a * b * (1 + x)));
        eq!(
            e!(2 ^ (1 / 2) + 2).factor_out(),
            e!(2 ^ (1 / 2) * (1 + 2 ^ (1 / 2)))
        );
        eq!(
            e!(a * b * x + a * c * x + b * c * x).factor_out(),
            e!(x * (a * b + a * c + b * c))
        );
        eq!(e!(a / x + b / x), e!(a / x + b / x))
    }

    #[test]
    fn derivative() {
        let d = |e: Expr| e.derivative(e!(x)).reduce().rationalize().expand().factor_out();

        eq!(d(e!(x ^ 2)), e!(2 * x));
        eq!(d(e!(sin(x))), e!(cos(x)));
        eq!(d(e!(exp(x))), e!(exp(x)));
        eq!(d(e!(x * exp(x))), e!(exp(x) * (1 + x)));
        eq!(d(e!(ln(x))), e!(1 / x));
        eq!(d(e!(1/x)), e!(-1 / x^2));
        eq!(d(e!(tan(x))), e!(sec(x)^2));
        eq!(d(e!(arc_tan(x))), e!(1 / (1 + x^2)));
        eq!(d(e!(x * ln(x) * sin(x))), e!(ln(x)*sin(x) + sin(x) + ln(x)*x*cos(x)));
        //eq!(d(e!(x ^ 2)), e!(2 * x));
        //eq!(d(exp(e!(sin(x)))), exp(e!(x)));
    }
}
