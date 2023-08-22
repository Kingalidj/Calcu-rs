//! simple symbolic algebra system in rust

use calcrs_macros::*;

#[init_calcrs_macro_scope]
/// this is only used, because innert attribute macros are unstable
mod __ {

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

    use core::fmt::Debug;

    #[derive(Debug, Clone, PartialEq)]
    pub enum CalcrsType {
        BooleanAtom(BooleanAtom),
        BooleanTrue(BooleanTrue),
        BooleanFalse(BooleanFalse),
        And(And),
    }

    pub type Eval = CalcrsType;

    pub trait Basic: Debug + Clone + Into<CalcrsType> {
        fn eval(&self) -> CalcrsType {
            self.clone().into()
        }
    }
    pub trait Boolean: Basic {}
    pub trait Application: Basic {}
    pub trait BooleanFunc: Boolean + Application {}

    #[derive(Debug, Default, Clone, Copy, PartialEq)]
    pub struct BooleanAtom {}
    impl Basic for BooleanAtom {}
    impl Boolean for BooleanAtom {}

    impl From<BooleanAtom> for CalcrsType {
        fn from(value: BooleanAtom) -> Self {
            CalcrsType::BooleanAtom(value)
        }
    }

    #[derive(Debug, Clone, Default, Copy, PartialEq)]
    pub struct BooleanTrue {}
    impl Basic for BooleanTrue {}
    impl Boolean for BooleanTrue {}

    impl From<BooleanTrue> for CalcrsType {
        fn from(value: BooleanTrue) -> Self {
            CalcrsType::BooleanTrue(value)
        }
    }

    #[derive(Debug, Clone, Default, Copy, PartialEq)]
    pub struct BooleanFalse {}
    impl Basic for BooleanFalse {}
    impl Boolean for BooleanFalse {}

    impl From<BooleanFalse> for CalcrsType {
        fn from(value: BooleanFalse) -> Self {
            CalcrsType::BooleanFalse(value)
        }
    }

    #[derive(Debug, Clone, PartialEq)]
    pub struct And {
        left: Box<CalcrsType>,
        right: Box<CalcrsType>,
    }

    impl And {
        pub fn new(lhs: impl Boolean, rhs: impl Boolean) -> Self {
            And {
                left: Box::new(lhs.into()),
                right: Box::new(rhs.into()),
            }
        }
    }

    impl Basic for And {}
    impl Boolean for And {}
    impl Application for And {}
    impl BooleanFunc for And {}

    impl From<And> for CalcrsType {
        fn from(value: And) -> Self {
            CalcrsType::And(value)
        }
    }

    #[allow(non_upper_case_globals)]
    pub static False: BooleanFalse = BooleanFalse {};

    #[allow(non_upper_case_globals)]
    pub static True: BooleanTrue = BooleanTrue {};
}
