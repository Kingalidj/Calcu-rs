use std::{cell::RefCell, collections::HashMap, rc::Rc};

use derive_more::Display;

use crate::binop::{Add, Mul};
use crate::boolean::Boolean;
use crate::numbers::Number;
use crate::traits::CalcursType;

pub type PTR<T> = Box<T>;
pub type SubsDict = Rc<RefCell<HashMap<Variable, Base>>>;

#[derive(Debug, Clone, Hash, PartialEq, Eq, Display)]
pub struct Base {
    pub kind: BasicKind,
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

impl From<BasicKind> for Base {
    fn from(value: BasicKind) -> Self {
        Base { kind: value }
    }
}

#[derive(Debug, Clone, Hash, PartialEq, Eq, Display)]
pub enum BasicKind {
    Var(Variable),
    Boolean(Boolean),
    Number(Number),

    Add(Add),
    Mul(Mul),

    Dummy,
}

impl BasicKind {
    pub fn subs(self, dict: SubsDict) -> Self {
        match self {
            BasicKind::Var(ref v) => {
                if dict.borrow().contains_key(&v) {
                    let basic = dict.borrow_mut().remove(&v).unwrap();
                    basic.kind
                } else {
                    self
                }
            }
            BasicKind::Boolean(b) => BasicKind::Boolean(b.kind.subs(dict).into()),
            BasicKind::Dummy => self,
            _ => todo!(), // BK::Number(_) => todo!(),
        }
    }

    pub fn simplify(self) -> BasicKind {
        match self {
            BasicKind::Boolean(b) => BasicKind::Boolean(b.kind.simplify().into()),
            BasicKind::Dummy | BasicKind::Var(_) => self,
            _ => todo!(), // BK::Number(_) => todo!(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Display)]
pub struct Variable {
    pub name: String,
}

impl<T: Into<String>> From<T> for Variable {
    fn from(value: T) -> Self {
        Variable { name: value.into() }
    }
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
        BasicKind::Var(self).into()
    }
}
