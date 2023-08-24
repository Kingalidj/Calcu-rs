//! simple symbolic algebra system in rust

use calcurs_internals::Inherited;
use core::fmt::Debug;
use std::ops;

use calcurs_macros::*;

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

impl Debug for Base {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Base {{...}}")
    }
}

macro_rules! base {
    ($($field:ident = $value:expr),* $(,)?) => {
        Base {
            $(
                $field: $value,
            )*
        ..Base::new()
    }
}
}

#[derive(Debug, Clone, PartialEq)]
pub enum CalcursType {
    BooleanTrue(BooleanTrue),
    BooleanFalse(BooleanFalse),
    And(And),
}

// macro_rules! map_calcurs_type {
//     ($e: expr, $func: expr) => {{
//         use CalcursType::*;
//         match $e {
//             BooleanTrue(ref x) => $func(x),
//             BooleanFalse(ref x) => $func(x),
//             And(ref x) => $func(x),
//         }
//     }};
// }

impl Inherited<Base> for CalcursType {
    fn base(&self) -> &Base {
        use CalcursType::*;

        match self {
            BooleanTrue(x) => x.base(),
            BooleanFalse(x) => x.base(),
            And(x) => x.base(),
        }
    }
}

pub type Eval = CalcursType;

#[init_calcurs_macro_scope]
mod __ {
    use std::any::Any;

    #[calcurs_base]
    #[derive(Default, Clone, Copy, PartialEq)]
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

    pub trait DynBasic: Debug {
        fn dyn_clone(&self) -> Box<dyn Basic>;
        fn calcrs_type(&self) -> CalcursType;
        fn as_any(&self) -> &dyn Any;
        fn dyn_eq(&self, _: &dyn Basic) -> bool;
    }

    impl<T: Basic + Clone + Into<CalcursType> + PartialEq + 'static> DynBasic for T {
        fn dyn_clone(&self) -> Box<dyn Basic> {
            Box::new(self.clone())
        }

        fn calcrs_type(&self) -> CalcursType {
            self.clone().into()
        }

        fn as_any(&self) -> &dyn Any {
            self
        }

        fn dyn_eq(&self, other: &dyn Basic) -> bool {
            DynBasic::as_any(other)
                .downcast_ref::<T>()
                .map_or(false, |x| self == x)
        }
    }

    impl Clone for Box<dyn Basic> {
        fn clone(&self) -> Box<dyn Basic> {
            self.dyn_clone()
        }
    }

    impl Into<CalcursType> for Box<dyn Basic> {
        fn into(self) -> CalcursType {
            self.calcrs_type()
        }
    }

    impl PartialEq for dyn Basic {
        fn eq(&self, other: &dyn Basic) -> bool {
            self.dyn_eq(other)
        }
    }

    // pub trait Basic: Debug + Clone + Into<CalcursType> + Inherited<Base> {
    pub trait Basic: DynBasic + Inherited<Base> {
        fn eval_impl(&self) -> Box<dyn Basic> {
            self.dyn_clone()
        }

        fn eval(&self) -> CalcursType {
            self.eval_impl().into()
        }
    }

    pub trait DynBoolean: Debug {
        fn dyn_clone(&self) -> Box<dyn Boolean>;
        fn calcrs_type(&self) -> CalcursType;
        fn as_any(&self) -> &dyn Any;
        fn dyn_eq(&self, _: &dyn Boolean) -> bool;
    }

    impl<T: Boolean + Clone + Into<CalcursType> + PartialEq + 'static> DynBoolean for T {
        fn dyn_clone(&self) -> Box<dyn Boolean> {
            Box::new(self.clone())
        }

        fn calcrs_type(&self) -> CalcursType {
            self.clone().into()
        }

        fn as_any(&self) -> &dyn Any {
            self
        }

        fn dyn_eq(&self, other: &dyn Boolean) -> bool {
            DynBoolean::as_any(other)
                .downcast_ref::<T>()
                .map_or(false, |x| self == x)
        }
    }

    impl Clone for Box<dyn Boolean> {
        fn clone(&self) -> Box<dyn Boolean> {
            DynBoolean::dyn_clone(self.as_ref())
        }
    }

    impl PartialEq for dyn Boolean {
        fn eq(&self, other: &dyn Boolean) -> bool {
            DynBoolean::dyn_eq(self, other)
        }
    }

    impl Into<CalcursType> for Box<dyn Boolean> {
        fn into(self) -> CalcursType {
            DynBoolean::calcrs_type(self.as_ref())
        }
    }

    // #[calcurs_trait(is_boolean = true)]
    pub trait Boolean: DynBoolean + Basic {
        fn eval_impl(&self) -> Box<dyn Boolean> {
            DynBoolean::dyn_clone(self)
        }
    }
    pub trait Application: Basic {}
    pub trait BooleanAtom: Boolean {}
    pub trait BooleanFunc: BooleanAtom + Application {}

    #[calcurs_type]
    #[derive(Debug, Clone, Default, Copy, PartialEq)]
    pub struct BooleanTrue {}
    impl Basic for BooleanTrue {}
    impl Boolean for BooleanTrue {}

    impl BooleanTrue {
        pub const fn new() -> Self {
            let base = base!(is_boolean = true, is_atom = true, is_negative = Some(false));
            BooleanTrue { base }
        }
    }

    impl From<BooleanTrue> for CalcursType {
        fn from(value: BooleanTrue) -> Self {
            CalcursType::BooleanTrue(value)
        }
    }

    #[calcurs_type]
    #[derive(Debug, Clone, Default, Copy, PartialEq)]
    pub struct BooleanFalse {}
    impl Basic for BooleanFalse {}
    impl Boolean for BooleanFalse {}

    impl BooleanFalse {
        pub const fn new() -> Self {
            let base = base!(is_boolean = true, is_atom = true, is_negative = Some(true));
            BooleanFalse { base }
        }
    }

    impl From<BooleanFalse> for CalcursType {
        fn from(value: BooleanFalse) -> Self {
            CalcursType::BooleanFalse(value)
        }
    }

    #[calcurs_type]
    #[derive(Debug, Clone)]
    pub struct And {
        left: Box<dyn Boolean>,
        right: Box<dyn Boolean>,
    }

    impl PartialEq for And {
        fn eq(&self, other: &And) -> bool {
            self.left.as_ref() == other.left.as_ref() && self.right.as_ref() == other.right.as_ref()
        }
    }

    impl And {
        pub fn new(lhs: impl Boolean, rhs: impl Boolean) -> Self {
            And {
                base: Default::default(),
                left: DynBoolean::dyn_clone(&rhs),
                right: DynBoolean::dyn_clone(&lhs),
            }
        }
    }

    impl Basic for And {
        fn eval(&self) -> CalcursType {
            Basic::eval_impl(Boolean::eval_impl(self).as_ref()).into()
        }
    }

    impl Boolean for And {
        fn eval_impl(&self) -> Box<dyn Boolean> {
            let lhs = Boolean::eval_impl(self.left.as_ref());
            let rhs = Boolean::eval_impl(self.right.as_ref());

            match lhs == rhs {
                true => DynBoolean::dyn_clone(&True),
                false => DynBoolean::dyn_clone(&False),
            }
        }
    }

    impl BooleanAtom for And {}
    impl Application for And {}
    impl BooleanFunc for And {}

    impl From<And> for CalcursType {
        fn from(value: And) -> Self {
            CalcursType::And(value)
        }
    }
}

impl<U: Boolean> ops::BitAnd<U> for BooleanTrue {
    type Output = And;

    fn bitand(self, rhs: U) -> Self::Output {
        And::new(self, rhs)
    }
}

impl<U: Boolean> ops::BitAnd<U> for BooleanFalse {
    type Output = And;

    fn bitand(self, rhs: U) -> Self::Output {
        And::new(self, rhs)
    }
}

impl<U: Boolean> ops::BitAnd<U> for And {
    type Output = And;

    fn bitand(self, rhs: U) -> Self::Output {
        And::new(self, rhs)
    }
}

#[allow(non_upper_case_globals)]
pub static False: BooleanFalse = BooleanFalse::new();

#[allow(non_upper_case_globals)]
pub static True: BooleanTrue = BooleanTrue::new();

#[cfg(test)]
mod test {

    use crate::*;

    #[test]
    fn boolean() {
        assert_eq!(And::new(True, False), True & False);
        assert_eq!(CalcursType::from(False), (True & False).eval());
    }
}
