use std::{cell::RefCell, collections::HashMap, rc::Rc};

use boolean::Boolean;
use derive_more::Display;
use numbers::Number;

mod boolean;
mod numbers;

pub mod prelude {
    pub use crate::boolean::*;
    pub use crate::numbers::*;
    pub use crate::*;
}

#[inline(always)]
pub const fn is_same<T: CalcursType, U: CalcursType>() -> bool {
    T::ID as u32 == U::ID as u32
}

#[inline(always)]
pub const fn cast_ref<'a, T: CalcursType, U: CalcursType>(r#ref: &'a T) -> Option<&'a U> {
    if is_same::<T, U>() {
        let ptr = r#ref as *const T as *const U;
        let cast = unsafe { &*ptr };
        Some(cast)
    } else {
        None
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(u32)]
pub enum TypeID {
    Basic,
    Variable,

    Boolean,
    True,
    False,
    And,
    Or,
    Not,

    #[default]
    Dummy,
}

pub type PTR<T> = Box<T>;
pub type ArgSet<T> = Vec<T>;

pub type SubsDict = Rc<RefCell<HashMap<Variable, Basic>>>;

pub trait CalcursType: Clone {
    const ID: TypeID;

    fn to_basic(self) -> Basic;
}

macro_rules! early_ret {
    ($e: expr) => {
        if let Some(ret) = $e {
            return Some(ret);
        }
    };
}

// should be completely optimized away => becomes same as just pattern matching
macro_rules! get_ref_impl {
    () => {
        #[inline(always)]
        pub const fn get_ref<T: CalcursType>(&self) -> Option<&T> {
            cast_ref::<Self, T>(self)
        }
    };

    ($($x: ident)+) => {
        #[inline(always)]
        pub const fn get_ref<T: CalcursType>(&self) -> Option<&T> {
            early_ret!(cast_ref::<Self, T>(self));
            $( early_ret!(self.$x.get_ref::<T>()); )+
            None
        }
    }
}

pub(crate) use early_ret;
pub(crate) use get_ref_impl;

#[derive(Debug, Clone, Hash, PartialEq, Eq, Display)]
pub struct Basic {
    pub kind: BasicKind,
}

impl Basic {
    get_ref_impl!(kind);

    pub fn subs<VAR: Into<Variable>, T: CalcursType>(self, var: VAR, b: T) -> Self {
        let dict = Rc::new(RefCell::new([(var.into(), b.to_basic())].into()));
        self.kind.subs(dict).into()
    }

    pub const fn is<T: CalcursType>(&self) -> bool {
        self.get_ref::<T>().is_some()
    }

    pub fn simplify(self) -> Basic {
        self.kind.simplify().into()
    }
}

impl CalcursType for Basic {
    const ID: TypeID = TypeID::Basic;

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

    Dummy,
}

impl BasicKind {
    #[inline(always)]
    pub const fn get_ref<T: CalcursType>(&self) -> Option<&T> {
        use BasicKind as BK;
        match self {
            BK::Var(b) => b.get_ref::<T>(),
            BK::Boolean(b) => b.get_ref::<T>(),
            BK::Dummy => None,
            _ => todo!(), // BK::Number(_) => todo!(),
        }
    }

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
}

impl CalcursType for Variable {
    const ID: TypeID = TypeID::Variable;

    fn to_basic(self) -> Basic {
        BasicKind::Var(self).into()
    }
}

impl Variable {
    get_ref_impl!();
}
