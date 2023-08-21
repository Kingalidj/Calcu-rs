//! simple symbolic algebra system in rust

use core::fmt::Debug;

#[derive(Debug, Default, Clone, Copy, PartialEq)]
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
    is_matmul: Option<bool>,
    is_real: Option<bool>,
    is_zero: Option<bool>,
    is_negative: Option<bool>,
    is_commutative: Option<bool>,
}

impl Base {
    pub const fn new() -> Self {
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
            is_matmul: None,
            is_real: None,
            is_zero: None,
            is_negative: None,
            is_commutative: None,
        }
    }
}

#[derive(Debug)]
pub struct Simplified(Box<dyn CalcrsType>);

impl Simplified {
    pub fn new<T: Basic>(val: T) -> Self {
        Simplified(val.into())
    }
}

#[derive(Debug)]
pub struct Eval(Box<dyn CalcrsType>);

impl Eval {
    pub fn new<T: Basic>(val: T) -> Self {
        Eval(val.into())
    }
}

pub trait CalcrsType: Debug {}
impl<T: Basic> CalcrsType for T {}
impl<T: Basic> From<T> for Box<dyn CalcrsType> {
    fn from(value: T) -> Self {
        Box::new(value)
    }
}

pub trait Basic: Copy + Clone + Debug + 'static {
    fn eval(&self) -> Eval {
        Eval::new(*self)
    }
}
pub trait Boolean: Basic {}
pub trait Application: Basic {}
pub trait BooleanFunc: Boolean + Application {}

#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub struct BooleanAtom {}
impl Basic for BooleanAtom {}
impl Boolean for BooleanAtom {}

#[derive(Debug, Clone, Default, Copy, PartialEq)]
pub struct BooleanTrue {}
impl Basic for BooleanTrue {}
impl Boolean for BooleanTrue {}

#[derive(Debug, Clone, Default, Copy, PartialEq)]
pub struct BooleanFalse {}
impl Basic for BooleanFalse {}
impl Boolean for BooleanFalse {}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct And<T: Basic, U: Basic> {
    left: T,
    right: U,
}
impl<T: Basic, U: Basic> Basic for And<T, U> {
    fn eval(&self) -> Eval {
        let _ = self.left.eval();
        let rhs = self.left.eval();
        rhs
    }
}
impl<T: Basic, U: Basic> Boolean for And<T, U> {}
impl<T: Basic, U: Basic> Application for And<T, U> {}
impl<T: Basic, U: Basic> BooleanFunc for And<T, U> {}

#[allow(non_upper_case_globals)]
pub static False: BooleanFalse = BooleanFalse {};

#[allow(non_upper_case_globals)]
pub static True: BooleanTrue = BooleanTrue {};
