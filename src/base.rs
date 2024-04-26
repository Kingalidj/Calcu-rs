use std::{fmt, ops};

use crate::{
    numeric::Numeric,
    operator::{Add, Div, Mul, Pow, Sub},
    pattern::{Item, Pattern, Pattern2},
};

/// implemented by every symbolic math type
pub trait CalcursType: Clone + fmt::Debug {
    fn desc(&self) -> Pattern;
    fn base(self) -> Base;

    fn match_local(&self, pattern: Pattern2) -> bool {
        self.desc().to_item().contains(pattern.get_item())
    }
}

pub type PTR<T> = Box<T>;

#[derive(Clone, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub enum Base {
    Symbol(Symbol),
    Numeric(Numeric),

    Add(Add),
    Mul(Mul),
    Pow(PTR<Pow>),
}

impl Base {
    pub fn pow(self, other: impl CalcursType) -> Base {
        Pow::pow(self, other).base()
    }
}

impl CalcursType for Base {
    fn desc(&self) -> Pattern {
        use Base as B;
        match self {
            B::Symbol(s) => s.desc(),
            B::Numeric(n) => n.desc(),
            B::Add(add) => add.desc(),
            B::Mul(mul) => mul.desc(),
            B::Pow(pow) => pow.desc(),
        }
    }
    #[inline(always)]
    fn base(self) -> Self {
        self
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
    fn desc(&self) -> Pattern {
        Pattern::Itm(Item::Symbol)
    }

    #[inline(always)]
    fn base(self) -> Base {
        Base::Symbol(self)
    }

}
impl CalcursType for &Symbol {
    fn desc(&self) -> Pattern {
        Pattern::Itm(Item::Symbol)
    }

    #[inline(always)]
    fn base(self) -> Base {
        panic!("only used for derivative")
    }
}

impl ops::Add for Base {
    type Output = Base;

    fn add(self, rhs: Self) -> Self::Output {
        Add::add(self, rhs)
    }
}
impl ops::AddAssign for Base {
    fn add_assign(&mut self, rhs: Self) {
        unsafe {
            // lhs = { 0 }
            // lhs = self
            // self = lhs + rhs
            let mut lhs: Base = std::mem::zeroed();
            std::mem::swap(self, &mut lhs);
            *self = Add::add(lhs, rhs);
        }
    }
}
impl ops::Sub for Base {
    type Output = Base;

    fn sub(self, rhs: Self) -> Self::Output {
        Sub::sub(self, rhs)
    }
}
impl ops::SubAssign for Base {
    fn sub_assign(&mut self, rhs: Self) {
        *self = Sub::sub(self.clone(), rhs);
    }
}
impl ops::Mul for Base {
    type Output = Base;

    fn mul(self, rhs: Self) -> Self::Output {
        Mul::mul(self, rhs)
    }
}
impl ops::MulAssign for Base {
    fn mul_assign(&mut self, rhs: Self) {
        // self *= rhs => self = self * rhs
        unsafe {
            // lhs = { 0 }
            // lhs = self
            // self = lhs * rhs
            let mut lhs = std::mem::zeroed();
            std::mem::swap(self, &mut lhs);
            *self = Mul::mul(lhs, rhs);
        }
    }
}
impl ops::Neg for Base {
    type Output = Base;

    fn neg(self) -> Self::Output {
        crate::rational::Rational::minus_one().base() * self
    }
}
impl ops::Div for Base {
    type Output = Base;

    fn div(self, rhs: Self) -> Self::Output {
        Div::div(self, rhs)
    }
}
impl ops::DivAssign for Base {
    fn div_assign(&mut self, rhs: Self) {
        unsafe {
            // lhs = { 0 }
            // lhs = self
            // self = lhs / rhs
            let mut lhs = std::mem::zeroed();
            std::mem::swap(self, &mut lhs);
            *self = Div::div(lhs, rhs);
        }
    }
}

impl<T: Into<String>> From<T> for Symbol {
    fn from(value: T) -> Self {
        Symbol { name: value.into() }
    }
}

impl fmt::Display for Base {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use Base as B;
        match self {
            B::Symbol(v) => write!(f, "{v}"),
            B::Numeric(n) => write!(f, "{n}"),

            B::Add(a) => write!(f, "{a}"),
            B::Mul(m) => write!(f, "{m}"),
            B::Pow(p) => write!(f, "{p}"),
        }
    }
}
impl fmt::Display for Symbol {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name)
    }
}
impl fmt::Debug for Symbol {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name)
    }
}
impl fmt::Debug for Base {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use Base as B;
        match self {
            B::Symbol(v) => write!(f, "{:?}", v),
            B::Numeric(n) => write!(f, "{:?}", n),

            B::Add(a) => write!(f, "{:?}", a),
            B::Mul(m) => write!(f, "{:?}", m),
            B::Pow(p) => write!(f, "{:?}", p),
        }
    }
}
