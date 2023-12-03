use std::{
    collections::HashMap,
    fmt::{Debug, Display},
    ops,
};

use crate::{
    derivative::Derivative,
    numeric::Numeric,
    operator::{Add, Div, Mul, Pow, Sub},
};

/// implemented by every symbolic math type
pub trait CalcursType: Clone + Debug {
    fn base(self) -> Base;
}

pub type PTR<T> = Box<T>;
pub type SubsDict<'a> = HashMap<String, Base>;

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

    pub fn subs(&self, dict: &SubsDict) -> Base {
        if let Some(key) = dict.get(&self.name) {
            key.clone()
        } else {
            self.clone().base()
        }
    }
}

impl Display for Symbol {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name)
    }
}

impl CalcursType for Symbol {
    #[inline(always)]
    fn base(self) -> Base {
        Base::Symbol(self).base()
    }
}

#[derive(Debug, Clone, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub enum Base {
    Symbol(Symbol),
    Numeric(Numeric),

    Add(Add),
    Mul(Mul),
    Pow(PTR<Pow>),

    Derivative(Derivative),
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
        Rational::from($int).base()
    };

    ($val: literal / $denom: literal) => {
        Rational::from(($val, $denom)).base()
    };

    (v: $var: ident) => {
        Symbol::new(stringify!($var)).base()
    };
}

impl Display for Base {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use Base as B;
        match self {
            B::Symbol(v) => write!(f, "{v}"),
            B::Numeric(n) => write!(f, "{n}"),

            B::Add(a) => write!(f, "{a}"),
            B::Mul(m) => write!(f, "{m}"),
            B::Pow(p) => write!(f, "{p}"),
            B::Derivative(d) => write!(f, "{:?}", d),
        }
    }
}

impl CalcursType for Base {
    #[inline(always)]
    fn base(self) -> Self {
        self
    }
}

impl Base {
    pub fn subs(self, dict: &SubsDict) -> Base {
        match self {
            Base::Symbol(s) => s.subs(dict),
            Base::Numeric(n) => n.subs(dict).base(),
            Base::Add(a) => a.subs(dict),
            Base::Mul(m) => m.subs(dict),
            Base::Pow(p) => p.subs(dict),
            Base::Derivative(d) => d.subs(dict),
        }
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

// TODO: support add assign without clone
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

// TODO: support mul assign without clone
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

#[cfg(test)]
mod base_test {

    use crate::prelude::*;
    use pretty_assertions::assert_eq;
    use std::collections::HashMap;

    #[test]
    fn subs() {
        let dict = HashMap::from([("x".to_owned(), base!(2))]);

        let expr = base!(v: x) ^ base!(2) + base!(4) * base!(v: x) + base!(3);
        assert_eq!(expr.subs(&dict), base!(15));
    }
}
