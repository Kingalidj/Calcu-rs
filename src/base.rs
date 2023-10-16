use std::ops;
use std::{cell::RefCell, collections::HashMap, rc::Rc};

use derive_more::Display;

use crate::binop::{Add, Mul};
use crate::boolean::Boolean;
use crate::numeric::{Integer, Number};
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

    pub fn simplify(self) -> Self {
        self
    }

    pub fn is_zero(&self) -> bool {
        false
    }
}

impl CalcursType for Variable {
    fn base(self) -> Base {
        BaseKind::Var(self).into()
    }
}

impl BaseKind {
    pub fn subs(self, dict: SubsDict) -> Self {
        match self {
            BaseKind::Var(ref v) => {
                if dict.borrow().contains_key(v) {
                    let basic = dict.borrow_mut().remove(v).unwrap();
                    basic.kind
                } else {
                    self
                }
            }
            BaseKind::Boolean(b) => b.kind.subs(dict).base().kind,
            BaseKind::Dummy => self,
            _ => todo!(), // BK::Number(_) => todo!(),
        }
    }

    pub fn simplify(self) -> BaseKind {
        match self {
            BaseKind::Boolean(b) => b.kind.simplify().base().kind,
            BaseKind::Dummy | BaseKind::Var(_) => self,
            _ => todo!(), // BK::Number(_) => todo!(),
        }
    }
}

#[derive(Debug, Clone, Hash, PartialEq, Eq, Display)]
pub enum BaseKind {
    Var(Variable),
    Boolean(Boolean),
    Number(Number),

    Add(Add),
    Mul(Mul),

    Dummy,
}

impl From<BaseKind> for Base {
    fn from(value: BaseKind) -> Self {
        Base { kind: value }
    }
}

#[derive(Debug, Clone, Hash, PartialEq, Eq, Display)]
pub struct Base {
    pub kind: BaseKind,
}

impl Base {
    pub fn subs<VAR: Into<Variable>, T: CalcursType>(self, var: VAR, b: T) -> Self {
        let dict = Rc::new(RefCell::new([(var.into(), b.base())].into()));
        self.kind.subs(dict).into()
    }

    pub fn simplify(self) -> Base {
        self.kind.simplify().into()
    }
}

impl CalcursType for Base {
    fn base(self) -> Self {
        self
    }
}

impl<T: Into<String>> From<T> for Variable {
    fn from(value: T) -> Self {
        Variable { name: value.into() }
    }
}

impl ops::Add for Base {
    type Output = Base;

    fn add(self, rhs: Self) -> Self::Output {
        Add::add(self, rhs)
    }
}

impl ops::Mul for Base {
    type Output = Base;

    fn mul(self, rhs: Self) -> Self::Output {
        Mul::mul(self, rhs)
    }
}

impl ops::Sub for Base {
    type Output = Base;

    fn sub(self, rhs: Self) -> Self::Output {
        Add::add(self, Mul::mul(Integer::new(-1), rhs))
    }
}
