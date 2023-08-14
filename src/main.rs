#![allow(dead_code)]

use std::ops;

#[derive(Debug, Clone, Copy, PartialEq)]
//TODO: as derive macro??
pub struct Base {
    is_number: bool,
    is_atom: bool,
    is_symbol: bool,
    is_function: bool,
    is_add: bool,
    is_mul: bool,
    is_pow: bool,
    is_float: bool,
    is_rational: bool,
    is_integer: bool,
    is_numbersymbol: bool,
    is_order: bool,
    is_derivative: bool,
    is_piecewise: bool,
    is_poly: bool,
    is_algebraicnumber: bool,
    is_relational: bool,
    is_equality: bool,
    is_boolean: bool,
    is_not: bool,
    is_matrix: bool,
    is_vector: bool,
    is_point: bool,
    is_matadd: bool,
    is_matmul: bool,
    is_real: bool,
    is_zero: bool,
    is_negative: bool,
    is_commutative: bool,
}

const DEFAULT_BASE: Base = Base {
    is_number: false,
    is_atom: false,
    is_symbol: false,
    is_function: false,
    is_add: false,
    is_mul: false,
    is_pow: false,
    is_float: false,
    is_rational: false,
    is_integer: false,
    is_numbersymbol: false,
    is_order: false,
    is_derivative: false,
    is_piecewise: false,
    is_poly: false,
    is_algebraicnumber: false,
    is_relational: false,
    is_equality: false,
    is_boolean: false,
    is_not: false,
    is_matrix: false,
    is_vector: false,
    is_point: false,
    is_matadd: false,
    is_matmul: false,
    is_real: false,
    is_zero: false,
    is_negative: false,
    is_commutative: false,
};

impl Base {
    const fn default() -> Self {
        DEFAULT_BASE
    }
}

macro_rules! impl_basic {
    ($struct_name: ident) => {
        impl Basic for $struct_name {
            fn base(&self) -> Base {
                self.base
            }
        }
    };
}

macro_rules! base {
    ($($field:ident = $value:expr),* $(,)?) => {
        Base {
            $(
                $field: $value,
            )*
            ..Base::default()
        }
    }
}

pub trait Basic {
    fn base(&self) -> Base;
}

/// Instances of Application represent the result of applying an application of any type to any object.
pub trait Application: Basic {}

trait Boolean: Basic {}

// Define the BooleanAtom struct that holds either true or false
#[derive(Debug, Clone, Copy, PartialEq)]
struct BooleanAtom {
    base: Base,
    value: bool,
}

impl BooleanAtom {
    const fn new(value: bool) -> Self {
        let base = base!(is_commutative = true, is_boolean = true, is_atom = true);
        BooleanAtom { base, value }
    }
}

impl Boolean for BooleanAtom {}
impl_basic!(BooleanAtom);

#[derive(Debug, Clone, Copy, PartialEq)]
struct BitAnd<T, U> {
    base: Base,
    left: T,
    right: U,
}

impl<T, U> Boolean for BitAnd<T, U> {}
impl<T, U> Basic for BitAnd<T, U> {
    fn base(&self) -> Base {
        self.base
    }
}

impl<T, U> BitAnd<T, U>
where
    T: Boolean,
    U: Boolean,
{
    const fn new(left: T, right: U) -> Self {
        let base = base!(is_commutative = true, is_boolean = true);
        BitAnd { base, left, right }
    }
}

impl<U: Boolean> ops::BitAnd<U> for BooleanAtom {
    type Output = BitAnd<Self, U>;

    fn bitand(self, rhs: U) -> Self::Output {
        BitAnd::new(self, rhs)
    }
}

impl<T: Boolean, U: Boolean> ops::BitAnd<U> for BitAnd<T, U> {
    type Output = BitAnd<Self, U>;

    fn bitand(self, rhs: U) -> Self::Output {
        BitAnd::new(self, rhs)
    }
}

#[allow(non_upper_case_globals)]
const False: BooleanAtom = BooleanAtom::new(false);
#[allow(non_upper_case_globals)]
const True: BooleanAtom = BooleanAtom::new(true);

fn main() {
    let b1 = False;
    let a1 = BitAnd::new(b1, b1);
    let b2 = True;
    let a2 = BitAnd::new(a1, b2);
    println!("{:?}", a2);
}

#[cfg(test)]
mod test {
    use crate::*;

    #[test]
    fn bitand() {
        assert_eq!(False & True, BitAnd::new(False, True))
    }
}
