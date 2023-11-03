use lazy_static::lazy_static;

use crate::{
    base::Base,
    boolean::BooleanAtom,
    numeric::{Number, Rational},
    rational::NonZeroUInt,
};

// pub const FALSE: Base = BooleanAtom::False;

// pub const TRUE: Base = BooleanAtom::True;

pub const ONE: Number = Number::Rational(Rational {
    is_neg: false,
    numer: 1,
    denom: NonZeroUInt::new(1),
});

pub const MINUS_ONE: Number = Number::Rational(Rational {
    is_neg: true,
    numer: 1,
    denom: NonZeroUInt::new(1),
});

pub const ZERO: Number = Number::Rational(Rational {
    is_neg: false,
    numer: 0,
    denom: NonZeroUInt::new(1),
});

lazy_static! {
    pub static ref FALSE: Base = BooleanAtom::False.base();
    pub static ref TRUE: Base = BooleanAtom::True.base();
}
