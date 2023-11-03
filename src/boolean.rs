use derive_more::Display;

use crate::{
    base::Base,
    traits::{Bool, CalcursType},
};

#[derive(Debug, Clone, Hash, PartialEq, Eq, Display)]
pub enum BoolValue {
    True,
    False,
    Unknown,
}

impl BoolValue {
    pub fn is_true(&self) -> bool {
        matches!(self, BoolValue::True)
    }

    pub fn is_false(&self) -> bool {
        matches!(self, BoolValue::False)
    }
}

#[derive(Debug, Clone, Copy, Hash, PartialOrd, PartialEq, Eq, Display)]
pub enum BooleanAtom {
    True,
    False,
}

impl BooleanAtom {
    #[inline]
    pub fn base(self) -> Base {
        Base::BooleanAtom(self).base()
    }

    pub fn and_kind(&mut self, other: BooleanAtom) {
        if other.is_false() {
            *self = other;
        }
    }

    pub fn or_kind(&mut self, other: BooleanAtom) {
        if other.is_true() {
            *self = other;
        }
    }
}

impl Bool for BooleanAtom {
    fn bool_val(&self) -> BoolValue {
        match self {
            BooleanAtom::True => BoolValue::True,
            BooleanAtom::False => BoolValue::False,
        }
    }
}

impl From<bool> for BooleanAtom {
    fn from(value: bool) -> Self {
        match value {
            true => BooleanAtom::True,
            false => BooleanAtom::False,
        }
    }
}

impl CalcursType for BooleanAtom {
    #[inline]
    fn base(self) -> Base {
        Base::BooleanAtom(self).base()
    }
}
