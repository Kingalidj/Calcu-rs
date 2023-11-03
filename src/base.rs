use std::{cell::RefCell, collections::HashMap, rc::Rc};

use derive_more::Display;

use crate::{
    boolean::BooleanAtom,
    numeric::Number,
    operator::{Add, Div, Mul, Pow, Sub},
    traits::CalcursType,
};

pub type PTR<T> = Box<T>;
pub type SubsDict = Rc<RefCell<HashMap<Variable, Base>>>;

#[derive(Debug, Clone, PartialEq, PartialOrd, Ord, Eq, Hash, Display)]
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

#[derive(Debug, Clone, Hash, PartialEq, Eq, PartialOrd, Ord, Display)]
pub enum Base {
    Var(Variable),
    //TODO: move to numeric?
    BooleanAtom(BooleanAtom),
    Number(Number),
    Dummy,

    Add(PTR<Add>),
    Mul(PTR<Mul>),
    Pow(PTR<Pow>),
    //Or(Or),
    //And(And),
    //Not(Not),
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
        Infinity::default().base()
    };

    (nan) => {
        Undefined.base()
    };

    (false) => {
        FALSE.clone()
    };

    (true) => {
        TRUE.clone()
    };

    ($int: literal) => {
        Rational::int_num($int).base()
    };

    ($val: literal / $denom: literal) => {
        Rational::frac_num($val, $denom).base()
    };

    (v: $var: tt) => {
        Variable::new(stringify!($var)).base()
    };
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

impl std::ops::AddAssign for Base {
    fn add_assign(&mut self, rhs: Self) {
        *self = Add::add(self.clone(), rhs);
    }
}

impl std::ops::Sub for Base {
    type Output = Base;

    fn sub(self, rhs: Self) -> Self::Output {
        Sub::sub(self, rhs)
    }
}

impl std::ops::SubAssign for Base {
    fn sub_assign(&mut self, rhs: Self) {
        *self = Sub::sub(self.clone(), rhs);
    }
}

impl std::ops::Mul for Base {
    type Output = Base;

    fn mul(self, rhs: Self) -> Self::Output {
        Mul::mul(self, rhs)
    }
}

impl std::ops::MulAssign for Base {
    fn mul_assign(&mut self, rhs: Self) {
        *self = Mul::mul(self.clone(), rhs);
    }
}

impl std::ops::Div for Base {
    type Output = Base;

    fn div(self, rhs: Self) -> Self::Output {
        Div::div(self, rhs)
    }
}

impl std::ops::DivAssign for Base {
    fn div_assign(&mut self, rhs: Self) {
        *self = Div::div(self.clone(), rhs);
    }
}

impl std::ops::BitXor for Base {
    type Output = Base;

    fn bitxor(self, rhs: Self) -> Self::Output {
        Pow::pow(self, rhs)
    }
}

impl std::ops::BitXorAssign for Base {
    fn bitxor_assign(&mut self, rhs: Self) {
        *self = Pow::pow(self.clone(), rhs);
    }
}

//impl std::ops::BitOr for Base {
//    type Output = Base;
//
//    fn bitor(self, rhs: Self) -> Self::Output {
//        Or::or(self, rhs)
//    }
//}
//
//impl std::ops::BitAnd for Base {
//    type Output = Base;
//
//    fn bitand(self, rhs: Self) -> Self::Output {
//        And::and(self, rhs)
//    }
//}
//
//impl std::ops::Not for Base {
//    type Output = Base;
//
//    fn not(self) -> Self::Output {
//        Not::not(self)
//    }
//}
