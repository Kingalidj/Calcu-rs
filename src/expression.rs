use calcu_rs::rational::Rational;
use std::{fmt, ops};
use std::collections::HashSet;

use crate::numeric::{Float, Infinity};
use crate::{
    operator::{Sum, Quot, Prod, Pow, Diff},
    pattern::Item,
};

/// implemented by every symbolic math type
pub trait CalcursType: Clone + fmt::Debug + Into<Expr>{
    fn desc(&self) -> Item;
}

/// contains one or multiple expressions of type [Expr]
pub trait Construct: CalcursType {
    ///This operator returns false when [other] is identical
    /// to some complete subexpression of [self] and otherwise returns true
    fn free_of(&self, other: &Expr) -> bool;
    fn contains(&self, other: &Expr) -> bool {
        !self.free_of(other)
    }

    fn is_polynomial_in(&self, vars: &[Expr]) -> bool {
        todo!()
    }

    /// returns all variables in the expression (e.g x, but also sin(x))
    /// normally you should call [variables]
    fn all_variables(&self) -> Vec<Expr>;

    /// returns all unique variables in the expression (e.g x, but also sin(x))
    fn variables(&self) -> Vec<Expr> {
        let vars = self.variables();
        let unique_vars: HashSet<Expr> = vars.into_iter().collect();
        unique_vars.into_iter().collect()
    }

    /// returns a list of all operands of the main operator
    /// if no operator is present we return a single operand
    fn operands(&mut self) -> Vec<&mut Expr>;


    /// This function returns the ith operand of self
    //  For example, Operand(m ∗ x + b, 2) -> b
    #[inline]
    fn operand(&mut self, i: usize) -> Option<&mut Expr> {
        let ops = self.operands();

        if i < ops.len() {
            Some(self.operands().swap_remove(i))
        } else {
            None
        }
    }

    /// returns the first operand of the main operator if available
    #[inline]
    fn first(&mut self) -> Option<&mut Expr> {
        self.operand(0)
    }

    fn map(&mut self, function: impl Fn(&mut Expr)) {
        self.operands().iter_mut().for_each(|x| function(*x))
    }

    fn simplify(self) -> Expr;
}

pub type PTR<T> = Box<T>;

#[derive(Clone, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub enum Expr {
    Symbol(Symbol),
    Rational(Rational),
    Float(Float),
    Infinity(Infinity),

    Sum(Sum),
    Prod(Prod),
    Pow(PTR<Pow>),

    /// only used if the result is provenly undefined, e.g 0 / 0
    ///
    /// not the same as [f64::NAN]
    Undefined,
}

macro_rules! impl_from_for_expr {
    ($typ:ident) => {
        impl From<$typ> for Expr {
            #[inline(always)]
            fn from(value: $typ) -> Expr {
                Expr::$typ(value.into())
            }
        }
    };
}

impl_from_for_expr!(Float);
impl_from_for_expr!(Rational);
impl_from_for_expr!(Infinity);
impl_from_for_expr!(Sum);
impl_from_for_expr!(Prod);
impl_from_for_expr!(Pow);



impl Expr {
    pub fn pow(self, other: impl CalcursType) -> Expr {
        Pow::pow(self, other).into()
    }

    /// base(x) -> x, when x is a [Symbol], [Product] etc...
    /// base(x) -> x.operand(1), when x is a power,
    /// base(x) -> None, when x is a constant e.g [Rational], [Float], etc...
    pub fn base(&self) -> Option<&Expr> {
        use Expr as E;
        match self {
            E::Symbol(_)
            | E::Sum(_)
            | E::Prod(_) => Some(self),
            E::Pow(p) => Some(&p.base),
            E::Rational(_)
            | E::Float(_)
            | E::Infinity(_)
            | E::Undefined => None,
        }
    }

    /// base(x) -> 1, when x is a [Symbol], [Product] etc...
    /// base(x) -> x.operand(2), when x is a power,
    /// base(x) -> None, when x is a constant e.g [Rational], [Float], etc...
    pub fn exponent(&self) -> Option<&Expr> {
        use Expr as E;
        match self {
            E::Symbol(_)
            | E::Sum(_)
            | E::Prod(_) => Some(&Rational::ONE),
            E::Pow(p) => Some(&p.exponent),
            E::Rational(_)
            | E::Float(_)
            | E::Infinity(_)
            | E::Undefined => None,
        }
    }
}

impl CalcursType for Expr {
    fn desc(&self) -> Item {
        use Expr as E;
        match self {
            E::Symbol(s) => s.desc(),
            E::Rational(r) => r.desc(),
            E::Float(f) => f.desc(),
            E::Infinity(i) => i.desc(),
            E::Sum(a) => a.desc(),
            E::Prod(m) => m.desc(),
            E::Pow(p) => p.desc(),
            E::Undefined => Item::Undef,
        }
    }
}


impl Construct for Expr {
    fn free_of(&self, other: &Expr) -> bool {
        use Expr as E;
        match (self, other) {
            (E::Symbol(s1), E::Symbol(s2)) => s1 != s2,
            (E::Rational(r1), E::Rational(r2)) => r1 != r2,
            (E::Float(f1), E::Float(f2)) => f1 != f2,
            (E::Infinity(i1), E::Infinity(i2)) => i1 != i2,
            (E::Undefined, E::Undefined) => false,

            (E::Sum(sum), _) => sum.free_of(other),
            (E::Prod(prod), _) => prod.free_of(other),
            (E::Pow(pow), _) => pow.free_of(other),

            (E::Symbol(_), _)
            | (E::Rational(_), _)
            | (E::Float(_), _)
            | (E::Infinity(_), _)
            | (E::Undefined, _)
            => true,
        }
    }

    #[inline]
    fn operands(&mut self) -> Vec<&mut Expr> {
        use Expr as E;
        match self {
            E::Sum(sum) => sum.operands(),
            E::Prod(prod) => prod.operands(),
            E::Pow(pow) => pow.operands(),
            E::Symbol(_) | E::Rational(_) | E::Float(_) | E::Infinity(_) | E::Undefined => {
                vec![self]
            }
        }
    }

    #[inline]
    fn simplify(self) -> Expr {
        use Expr as E;
        match self {
            E::Symbol(_) | E::Rational(_) | E::Float(_) | E::Infinity(_) | E::Undefined => self,

            E::Sum(sum) => sum.simplify(),
            E::Prod(prod) => prod.simplify(),
            E::Pow(pow) => pow.simplify(),
        }
    }

    fn all_variables(&self) -> Vec<Expr> {
        match self {
            Expr::Rational(_)
            | Expr::Float(_)
            | Expr::Infinity(_)
            | Expr::Undefined => vec![],

            Expr::Symbol(_) => vec![self.clone()],
            Expr::Sum(sum) => sum.all_variables(),
            Expr::Prod(prod) => prod.all_variables(),
            Expr::Pow(pow) => pow.all_variables(),
        }
    }
}

#[derive(Clone, PartialEq, PartialOrd, Ord, Eq, Hash)]
pub struct Symbol {
    pub name: String,
}
impl Symbol {
    pub fn new<I: Into<String>>(name: I) -> Self {
        Self { name: name.into() }
    }
}

impl CalcursType for Symbol {
    fn desc(&self) -> Item {
        Item::Symbol
    }
}
impl From<Symbol> for Expr {
    #[inline(always)]
    fn from(value: Symbol) -> Expr {
        Expr::Symbol(value)
    }
}
impl CalcursType for &Symbol {
    fn desc(&self) -> Item {
        Item::Symbol
    }
}

impl From<&Symbol> for Expr {
    #[inline(always)]
    fn from(value: &Symbol) -> Expr {
        panic!("only used for derivative")
    }
}

impl ops::Add for Expr {
    type Output = Expr;

    fn add(self, rhs: Self) -> Self::Output {
        Sum::add(self, rhs)
    }
}
impl ops::AddAssign for Expr {
    fn add_assign(&mut self, rhs: Self) {
        unsafe {
            // lhs = { 0 }
            // lhs = self
            // self = lhs + rhs
            let mut lhs: Expr = std::mem::zeroed();
            std::mem::swap(self, &mut lhs);
            *self = Sum::add(lhs, rhs);
        }
    }
}
impl ops::Sub for Expr {
    type Output = Expr;

    fn sub(self, rhs: Self) -> Self::Output {
        Diff::sub(self, rhs)
    }
}
impl ops::SubAssign for Expr {
    fn sub_assign(&mut self, rhs: Self) {
        *self = Diff::sub(self.clone(), rhs);
    }
}
impl ops::Mul for Expr {
    type Output = Expr;

    fn mul(self, rhs: Self) -> Self::Output {
        Prod::mul(self, rhs)
    }
}
impl ops::MulAssign for Expr {
    fn mul_assign(&mut self, rhs: Self) {
        // self *= rhs => self = self * rhs
        unsafe {
            // lhs = { 0 }
            // lhs = self
            // self = lhs * rhs
            let mut lhs = std::mem::zeroed();
            std::mem::swap(self, &mut lhs);
            *self = Prod::mul(lhs, rhs);
        }
    }
}
impl ops::Neg for Expr {
    type Output = Expr;

    fn neg(self) -> Self::Output {
        Rational::MINUS_ONE * self
    }
}
impl ops::Div for Expr {
    type Output = Expr;

    fn div(self, rhs: Self) -> Self::Output {
        Quot::div(self, rhs)
    }
}
impl ops::DivAssign for Expr {
    fn div_assign(&mut self, rhs: Self) {
        unsafe {
            // lhs = { 0 }
            // lhs = self
            // self = lhs / rhs
            let mut lhs = std::mem::zeroed();
            std::mem::swap(self, &mut lhs);
            *self = Quot::div(lhs, rhs);
        }
    }
}

impl<T: Into<String>> From<T> for Symbol {
    fn from(value: T) -> Self {
        Symbol { name: value.into() }
    }
}

impl fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use Expr as E;
        match self {
            E::Symbol(v) => write!(f, "{v}"),
            E::Rational(r) => write!(f, "{r}"),
            E::Float(v) => write!(f, "{}", v.0),
            E::Infinity(i) => write!(f, "{i}"),
            E::Sum(a) => write!(f, "{a}"),
            E::Prod(m) => write!(f, "{m}"),
            E::Pow(p) => write!(f, "{p}"),

            E::Undefined => write!(f, "undefined"),
        }
    }
}
impl fmt::Display for Symbol {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name)
    }
}
impl fmt::Debug for Symbol {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name)
    }
}
impl fmt::Debug for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use Expr as E;
        match self {
            E::Symbol(v) => write!(f, "{:?}", v),
            E::Rational(r) => write!(f, "{:?}", r),
            E::Float(v) => write!(f, "{:?}", v),
            E::Infinity(i) => write!(f, "{:?}", i),

            E::Sum(a) => write!(f, "{:?}", a),
            E::Prod(m) => write!(f, "{:?}", m),
            E::Pow(p) => write!(f, "{:?}", p),

            E::Undefined => write!(f, "undefined"),
        }
    }
}
