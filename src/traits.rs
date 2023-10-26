use crate::{
    base::Base,
    boolean::{BoolValue, BooleanAtom},
    numeric::{Number, Rational, Sign},
};

pub trait CalcursType: Clone {
    fn base(self) -> Base;
}

pub trait Num {
    fn is_zero(&self) -> bool;
    fn is_one(&self) -> bool;
    fn sign(&self) -> Sign;

    fn is_pos(&self) -> bool {
        self.sign().is_pos()
    }

    fn is_neg(&self) -> bool {
        self.sign().is_neg()
    }
}

impl<N: Num> Bool for N {
    fn bool_val(&self) -> BoolValue {
        if self.is_zero() {
            BoolValue::False
        } else if self.is_pos() || self.is_neg() {
            BoolValue::True
        } else {
            BoolValue::Unknown
        }
    }
}

pub trait Bool {
    fn bool_val(&self) -> BoolValue;

    #[inline]
    fn is_true(&self) -> bool {
        self.bool_val().is_true()
    }

    #[inline]
    fn is_false(&self) -> bool {
        self.bool_val().is_false()
    }

    #[inline]
    fn to_num(&self) -> Number {
        match self.is_false() {
            true => Rational::int(0),
            false => Rational::int(1),
        }
    }

    fn to_bool(&self) -> Option<BooleanAtom> {
        match self.bool_val() {
            BoolValue::True => Some(BooleanAtom::True),
            BoolValue::False => Some(BooleanAtom::False),
            BoolValue::Unknown => None,
        }
    }
}
