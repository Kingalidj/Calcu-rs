use std::{
    cell::RefCell,
    collections::HashMap,
    fmt::{Debug, Display},
    ops,
    rc::Rc,
};

use crate::{
    numeric::{Number, Sign},
    operator::{Add, Div, Mul, Pow, Sub},
};

/// implemented by every symbolic math type
pub trait CalcursType: Clone + Debug + Display {
    fn base(self) -> Base;
}

/// implemented by every struct that represenets a numeric type
pub trait Num {
    fn is_zero(&self) -> bool;
    fn is_one(&self) -> bool;
    fn is_neg_one(&self) -> bool;
    fn sign(&self) -> Option<Sign>;

    fn is_pos(&self) -> bool {
        self.sign().map_or_else(|| false, |s| s.is_pos())
    }

    fn is_neg(&self) -> bool {
        self.sign().map_or_else(|| false, |s| s.is_neg())
    }
}

pub type PTR<T> = Box<T>;
pub type SubsDict = Rc<RefCell<HashMap<Symbol, Base>>>;

#[derive(Debug, Clone, PartialEq, PartialOrd, Ord, Eq, Hash)]
pub struct Symbol {
    pub name: String,
}

impl Symbol {
    pub fn new<I: Into<String>>(name: I) -> Self {
        Self { name: name.into() }
    }

    pub fn is_zero(&self) -> bool {
        false
    }
}

impl Display for Symbol {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name)
    }
}

impl CalcursType for Symbol {
    #[inline]
    fn base(self) -> Base {
        Base::Symbol(self).base()
    }
}

#[derive(Debug, Clone, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub enum Base {
    Symbol(Symbol),
    Number(Number),
    // Dummy,
    Add(Add),
    Mul(Mul),
    Pow(PTR<Pow>),
}

#[macro_export]
macro_rules! base {
    (pos_inf) => {
        Infinity::pos().base()
    };

    (neg_inf) => {
        Infinity::neg().base()
    };

    (inf) => {
        Infinity::pos().base()
    };

    (nan) => {
        Undefined.base()
    };

    ($int: literal) => {
        Rational::int_num($int).base()
    };

    ($val: literal / $denom: literal) => {
        Rational::frac_num($val, $denom).base()
    };

    (v: $var: tt) => {
        Symbol::new(stringify!($var)).base()
    };
}

impl Display for Base {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Base::Symbol(v) => write!(f, "{v}"),
            Base::Number(n) => write!(f, "{n}"),
            // Base::Dummy => write!(f, "Dummy"),
            Base::Add(a) => write!(f, "{a}"),
            Base::Mul(m) => write!(f, "{m}"),
            Base::Pow(p) => write!(f, "{p}"),
        }
    }
}

impl CalcursType for Base {
    #[inline]
    fn base(self) -> Self {
        self
    }
}

impl<T: Into<String>> From<T> for Symbol {
    fn from(value: T) -> Self {
        Symbol { name: value.into() }
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
        *self = Add::add(self.clone(), rhs);
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
        *self = Mul::mul(self.clone(), rhs);
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
        *self = Div::div(self.clone(), rhs);
    }
}

impl ops::BitXor for Base {
    type Output = Base;

    fn bitxor(self, rhs: Self) -> Self::Output {
        Pow::pow(self, rhs)
    }
}

impl ops::BitXorAssign for Base {
    fn bitxor_assign(&mut self, rhs: Self) {
        *self = Pow::pow(self.clone(), rhs);
    }
}
