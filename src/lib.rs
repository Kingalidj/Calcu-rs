use std::{cell::RefCell, collections::HashMap, rc::Rc};

use boolean::Boolean;
use derive_more::Display;
use numbers::{Add, Mul, Number};

mod boolean;
mod numbers;

pub mod prelude {
    pub use crate::boolean::*;
    pub use crate::numbers::*;
    pub use crate::*;
}

// #[inline(always)]
// pub const fn is_same<T: CalcursType, U: CalcursType>() -> bool {
//     T::ID as u32 == U::ID as u32
// }

// #[inline(always)]
// pub const fn cast_ref<'a, T: CalcursType, U: CalcursType>(r#ref: &'a T) -> Option<&'a U> {
//     if is_same::<T, U>() {
//         let ptr = r#ref as *const T as *const U;
//         let cast = unsafe { &*ptr };
//         Some(cast)
//     } else {
//         None
//     }
// }

pub type PTR<T> = Box<T>;
pub type ArgSet<T> = Vec<T>;

pub type SubsDict = Rc<RefCell<HashMap<Variable, Basic>>>;

pub trait CalcursType: Clone {
    fn to_basic(self) -> Basic;
}

#[derive(Debug, Clone, Hash, PartialEq, Eq, Display)]
pub struct Basic {
    pub kind: BasicKind,
}

impl Basic {
    pub fn subs<VAR: Into<Variable>, T: CalcursType>(self, var: VAR, b: T) -> Self {
        let dict = Rc::new(RefCell::new([(var.into(), b.to_basic())].into()));
        self.kind.subs(dict).into()
    }

    pub fn simplify(self) -> Basic {
        self.kind.simplify().into()
    }
}

impl CalcursType for Basic {
    fn to_basic(self) -> Self {
        self
    }
}

impl From<BasicKind> for Basic {
    fn from(value: BasicKind) -> Self {
        Basic { kind: value }
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

pub trait SimplifyAs<T> {
    fn simplify(self) -> T;
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
    fn to_basic(self) -> Basic {
        BasicKind::Var(self).into()
    }
}
