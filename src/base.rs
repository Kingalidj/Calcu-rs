use std::{cell::RefCell, collections::HashMap, rc::Rc};

use derive_more::Display;

use crate::boolean::BooleanAtom;
use crate::numeric::Number;
use crate::operator::{Add, And, Div, Mul, Not, Or, Pow, Sub};
use crate::traits::CalcursType;

pub type PTR<T> = Box<T>;
pub type SubsDict = Rc<RefCell<HashMap<Variable, Base>>>;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Display)]
pub struct Variable {
    pub name: String,
}

impl Variable {
    pub fn new<I: Into<String>>(name: I) -> Self {
        Self { name: name.into() }
    }

    pub fn is_zero(&self) -> bool {
        false
    }
}

impl CalcursType for Variable {
    #[inline]
    fn base(self) -> Base {
        Base::Var(self).base()
    }
}

#[derive(Debug, Clone, Hash, PartialEq, Eq, Display)]
pub enum Base {
    Var(Variable),
    BooleanAtom(BooleanAtom),
    Number(Number),
    Dummy,

    Add(PTR<Add>),

    Mul(PTR<Mul>),
    Pow(PTR<Pow>),
    Or(Or),
    And(And),
    Not(Not),
}

impl CalcursType for Base {
    #[inline]
    fn base(self) -> Self {
        self
    }
}

impl<T: Into<String>> From<T> for Variable {
    fn from(value: T) -> Self {
        Variable { name: value.into() }
    }
}

impl std::ops::Add for Base {
    type Output = Base;

    fn add(self, rhs: Self) -> Self::Output {
        Add::add(self, rhs)
    }
}

impl std::ops::Mul for Base {
    type Output = Base;

    fn mul(self, rhs: Self) -> Self::Output {
        Mul::mul(self, rhs)
    }
}

impl std::ops::Div for Base {
    type Output = Base;

    fn div(self, rhs: Self) -> Self::Output {
        Div::div(self, rhs)
    }
}

impl std::ops::Sub for Base {
    type Output = Base;

    fn sub(self, rhs: Self) -> Self::Output {
        Sub::sub(self, rhs)
    }
}

impl std::ops::BitXor for Base {
    type Output = Base;

    fn bitxor(self, rhs: Self) -> Self::Output {
        Pow::pow(self, rhs)
    }
}

impl std::ops::BitOr for Base {
    type Output = Base;

    fn bitor(self, rhs: Self) -> Self::Output {
        Or::or(self, rhs)
    }
}

impl std::ops::BitAnd for Base {
    type Output = Base;

    fn bitand(self, rhs: Self) -> Self::Output {
        And::and(self, rhs)
    }
}

impl std::ops::Not for Base {
    type Output = Base;

    fn not(self) -> Self::Output {
        Not::not(self)
    }
}
