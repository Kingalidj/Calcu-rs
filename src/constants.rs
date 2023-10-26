use lazy_static::lazy_static;

use crate::{
    base::Base,
    boolean::BooleanAtom,
    numeric::{Number, Rational},
};

// pub const FALSE: Base = BooleanAtom::False;

// pub const TRUE: Base = BooleanAtom::True;

lazy_static! {
    pub static ref ZERO: Number = Rational::int(0);
    pub static ref ONE: Number = Rational::int(1);
    pub static ref FALSE: Base = BooleanAtom::False.base();
    pub static ref TRUE: Base = BooleanAtom::True.base();
}
