use lazy_static::lazy_static;

use crate::{
    base::Base,
    boolean::BooleanAtom,
    numeric::{Integer, Number},
};

// pub const FALSE: Base = BooleanAtom::False;

// pub const TRUE: Base = BooleanAtom::True;

lazy_static! {
    pub static ref ZERO: Number = Integer::num(0).into();
    pub static ref ONE: Number = Integer::num(1).into();
    pub static ref FALSE: Base = BooleanAtom::False.base();
    pub static ref TRUE: Base = BooleanAtom::True.base();
}
