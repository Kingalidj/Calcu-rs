use std::{collections::HashMap, fmt, ops};

use crate::{
    derivative::Derivative,
    numeric::Numeric,
    operator::{Add, Div, Mul, Pow, Sub},
};

use crate::pattern as pat;

pub type PTR<T> = Box<T>;
pub type SubsDict<'a> = HashMap<String, Base>;

#[derive(Debug, Clone, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub enum Base {
    Symbol(Symbol),
    Numeric(Numeric),

    Add(Add),
    Mul(Mul),
    Pow(PTR<Pow>),

    Derivative(Derivative),
}

#[derive(Debug, Clone, PartialEq, PartialOrd, Ord, Eq, Hash)]
pub struct Symbol {
    pub name: String,
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

    pub fn pow(self, other: Self) -> Base {
        Pow::pow(self, other).base()
    }

    #[inline]
    pub fn desc(&self) -> pat::Pattern {
        use Base as B;
        match self {
            B::Symbol(s) => s.desc().into(),
            B::Numeric(n) => n.desc().into(),
            B::Add(add) => add.desc(),
            B::Mul(mul) => mul.desc(),
            B::Pow(pow) => pow.desc(),
            B::Derivative(_) => todo!(),
        }
    }
}

impl Symbol {
    pub fn new<I: Into<String>>(name: I) -> Self {
        Self { name: name.into() }
    }

    pub fn subs(&self, dict: &SubsDict) -> Base {
        if let Some(key) = dict.get(&self.name) {
            key.clone()
        } else {
            self.clone().base()
        }
    }

    pub const fn desc(&self) -> pat::Pattern {
        pat::Pattern::Itm(pat::Item::Symbol)
    }
}

/// implemented by every symbolic math type
pub trait CalcursType: Clone + fmt::Debug {
    fn base(self) -> Base;
}

impl CalcursType for Base {
    #[inline(always)]
    fn base(self) -> Self {
        self
    }
}

impl CalcursType for Symbol {
    #[inline(always)]
    fn base(self) -> Base {
        Base::Symbol(self).base()
    }
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
            B::Derivative(d) => write!(f, "{:?}", d),
        }
    }
}

impl fmt::Display for Symbol {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name)
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

        let expr = base!(v: x).pow(base!(2)) + base!(4) * base!(v: x) + base!(3);
        assert_eq!(expr.subs(&dict), base!(15));
    }
}

#[cfg(test)]
mod display {
    use crate::prelude::*;
    use pretty_assertions::assert_eq;
    use test_case::test_case;

    macro_rules! b {
        ($($x: tt)*) => {
            base!($($x)*)
        }
    }

    #[test_case(b!(1).pow(b!(2)), "1")]
    #[test_case(b!(1 / 2).pow(b!(2)), "1/4")]
    #[test_case(b!(1 / 3).pow(b!(1 / 100)), "(1/3)^(1/100)")]
    #[test_case(b!(10).pow(b!(10)) + b!(1 / 1000), "10000000000001 e-3")]
    #[test_case(b!(1 / 3).pow(b!(2 / 1000)), "(1/3)^(1/500)")]
    fn disp_fractions(exp: Base, res: &str) {
        let fmt = format!("{}", exp);
        assert_eq!(fmt, res);
        
    }
}
