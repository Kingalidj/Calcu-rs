use std::{fmt, slice};
use crate::{
    expr::Expr, polynomial, rational::{Int, Rational}, utils
};

use derive_more::{Debug, Display, From, Into, IsVariant, Unwrap, TryUnwrap };

pub(crate) type PTR<T> = std::rc::Rc<T>;

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[derive(Debug, From, IsVariant, Unwrap, TryUnwrap)]
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

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[derive(Debug, Display, From)]
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

}

impl fmt::Debug for Pow {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}^{:?}", self.args[0], self.args[1])
    }
}
