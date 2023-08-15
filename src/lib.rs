//! simple symbolic algebra system in rust

#![warn(missing_docs)]
#![allow(dead_code)]
use core::fmt;
use std::ops;

use calcurs_internals::Inherited;
use calcurs_macros::*;

#[derive(Clone, Copy, PartialEq)]
/// All calcu-rs types have the field { base: [Base] }
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

impl Base {
    const fn default() -> Self {
        Base {
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
        }
    }
}

impl fmt::Debug for Base {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Base {{...}}")
    }
}

/// a simple macro for generating [Base] structs with
///
/// # Examples
///
/// ```
/// let base = base!(is_atom = true, is_commutative = true);
/// ```

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

/// Implemented by all Calcurs types
pub trait Basic: Inherited<Base> {}

/// Instances of Application represent the result of applying an application of any type to any object
pub trait Application: Basic {}

/// Boolean function is a function that lives in a boolean space
pub trait BooleanFunc: Application + Boolean {}

/// A Boolean object is an object for which logic operations make sense
pub trait Boolean: Basic {}

/// Ambiguous Boolean: same structure as BooleanTrue / BooleanFalse
#[derive(Debug, Clone, Copy, PartialEq)]
#[inherit(Base)]
pub struct BooleanAtom {}

impl BooleanAtom {
    /// create a new [BooleanAtom]

    pub const fn new() -> Self {
        let base = base!(is_commutative = true, is_boolean = true, is_atom = true);
        BooleanAtom { base }
    }
}

impl Basic for BooleanAtom {}
impl Boolean for BooleanAtom {}

/// Calcurus version of [true], a singleton that can be accessed via [True]
#[derive(Debug, Clone, Copy, PartialEq)]
#[inherit(Base)]
pub struct BooleanTrue {}

impl Basic for BooleanTrue {}
impl Boolean for BooleanTrue {}

impl BooleanTrue {
    /// create a new [BooleanTrue]
    pub const fn new() -> Self {
        let base = base!(is_commutative = true, is_boolean = true, is_atom = true);
        BooleanTrue { base }
    }
}

impl From<BooleanTrue> for bool {
    fn from(_: BooleanTrue) -> Self {
        true
    }
}

/// Calcurus version of [false], a singleton that can be accessed via [False]
#[derive(Debug, Clone, Copy, PartialEq)]
#[inherit(Base)]
pub struct BooleanFalse {}

impl Basic for BooleanFalse {}
impl Boolean for BooleanFalse {}

impl BooleanFalse {
    /// create a new [BooleanFalse]
    pub const fn new() -> Self {
        let base = base!(is_commutative = true, is_boolean = true, is_atom = true);
        BooleanFalse { base }
    }
}

impl From<BooleanFalse> for bool {
    fn from(_: BooleanFalse) -> Self {
        false
    }
}

/// Logical AND function.
/// The [ops::BitAnd] operator is overloaded for convenience
#[inherit(Base)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct And<T, U> {
    left: T,
    right: U,
}

impl<T, U> Boolean for And<T, U> {}
impl<T, U> Application for And<T, U> {}
impl<T, U> BooleanFunc for And<T, U> {}
impl<T, U> Basic for And<T, U> {}

impl<T, U> And<T, U>
where
    T: Boolean,
    U: Boolean,
{
    /// create a new [And<T, U>]
    pub const fn new(left: T, right: U) -> Self {
        let base = base!(is_commutative = true, is_boolean = true);
        And { base, left, right }
    }
}

impl<U: Boolean> ops::BitAnd<U> for BooleanAtom {
    type Output = And<Self, U>;

    fn bitand(self, rhs: U) -> Self::Output {
        And::new(self, rhs)
    }
}

impl<U: Boolean> ops::BitAnd<U> for BooleanTrue {
    type Output = And<Self, U>;

    fn bitand(self, rhs: U) -> Self::Output {
        And::new(self, rhs)
    }
}

impl<U: Boolean> ops::BitAnd<U> for BooleanFalse {
    type Output = And<Self, U>;

    fn bitand(self, rhs: U) -> Self::Output {
        And::new(self, rhs)
    }
}

impl<T: Boolean, U: Boolean> ops::BitAnd<U> for And<T, U> {
    type Output = And<Self, U>;

    fn bitand(self, rhs: U) -> Self::Output {
        And::new(self, rhs)
    }
}

/// Calcurus version of [false]
#[allow(non_upper_case_globals)]
pub static False: BooleanFalse = BooleanFalse::new();
/// Calcurus version of [true]
#[allow(non_upper_case_globals)]
pub static True: BooleanTrue = BooleanTrue::new();

#[cfg(test)]
mod test {
    use crate::*;

    #[test]
    fn bitand() {
        assert_eq!(False & True, And::new(False, True));
        assert_eq!(False & True & True, And::new(And::new(False, True), True));
    }
}
