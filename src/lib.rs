//! simple symbolic algebra system in rust

use calcurs_internals::Inherited;
use const_default::ConstDefault;
use core::fmt::Debug;
use std::ops;

use calcurs_macros::*;

impl Base {
    pub const fn new() -> Self {
        <Self as ConstDefault>::DEFAULT
    }
}

impl Debug for Base {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.diff_debug(f)
    }
}

macro_rules! base {
    ($($field:ident $(= $value:expr)?),* $(,)?) => {
        Base {
            $(
                $field $(: $value)?,
            )*
        ..Self::new_base()
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum CalcursType {
    BooleanTrue(BooleanTrue),
    BooleanFalse(BooleanFalse),
    And(And),
}

pub type Eval = CalcursType;

#[init_calcurs_macro_scope]
mod scope {
    use std::any::Any;

    #[calcurs_base(const Base::new)]
    #[derive(ConstDefault, Default, Clone, Copy, PartialEq)]
    pub struct Base {
        pub is_number: bool,
        pub is_atom: bool,
        pub is_symbol: bool,
        pub is_function: bool,
        pub is_add: bool,
        pub is_mul: bool,
        pub is_pow: bool,
        pub is_float: bool,
        pub is_rational: bool,
        pub is_integer: bool,
        pub is_numbersymbol: bool,
        pub is_order: bool,
        pub is_derivative: bool,
        pub is_piecewise: bool,
        pub is_poly: bool,
        pub is_algebraicnumber: bool,
        pub is_relational: bool,
        pub is_equality: bool,
        pub is_boolean: bool,
        pub is_not: bool,
        pub is_matrix: bool,
        pub is_vector: bool,
        pub is_point: bool,
        pub is_matadd: bool,
        pub is_matmul: Option<bool>,
        pub is_real: Option<bool>,
        pub is_zero: Option<bool>,
        pub is_negative: Option<bool>,
        pub is_commutative: Option<bool>,
    }

    pub trait IsCalcursType {
        fn as_calcrs_type(&self) -> CalcursType;
    }

    impl<T> IsCalcursType for T
    where
        T: Basic + Clone + Into<CalcursType> + PartialEq + 'static,
    {
        fn as_calcrs_type(&self) -> CalcursType {
            self.clone().into()
        }
    }

    impl From<Box<dyn Basic>> for CalcursType {
        fn from(val: Box<dyn Basic>) -> Self {
            val.as_calcrs_type()
        }
    }

    #[dyn_trait]
    #[calcurs_trait()]
    pub trait Basic: Debug + IsCalcursType + Inherited<Base> {
        fn eval_impl(&self) -> Box<dyn Basic>
        where
            Self: Sized,
        {
            self.dyn_clone()
        }

        fn eval(&self) -> CalcursType {
            self.as_calcrs_type()
        }
    }

    #[dyn_trait]
    #[calcurs_trait(is_boolean = true)]
    pub trait Boolean: Basic {
        fn eval_impl(&self) -> &dyn Boolean {
            DynBoolean::as_obj(self)
        }
    }

    #[dyn_trait]
    #[calcurs_trait(is_atom = true)]
    pub trait BooleanAtom: Boolean {}

    #[dyn_trait]
    #[calcurs_trait(is_function = true)]
    pub trait Application: Basic {}

    #[dyn_trait]
    #[calcurs_trait()]
    pub trait BooleanFunc: Boolean + Application {}

    #[calcurs_type(BooleanAtom)]
    #[derive(Debug, Clone, Default, Copy, PartialEq)]
    pub struct BooleanTrue {}
    impl Basic for BooleanTrue {}
    impl Boolean for BooleanTrue {}

    impl BooleanTrue {
        pub const fn new() -> Self {
            let base = base!(is_negative = Some(false));
            BooleanTrue { base }
        }
    }

    impl From<BooleanTrue> for CalcursType {
        fn from(value: BooleanTrue) -> Self {
            CalcursType::BooleanTrue(value)
        }
    }

    impl From<BooleanTrue> for bool {
        fn from(_: BooleanTrue) -> bool {
            true
        }
    }

    #[calcurs_type(BooleanAtom)]
    #[derive(Debug, Clone, Default, Copy, PartialEq)]
    pub struct BooleanFalse {}
    impl Basic for BooleanFalse {}
    impl Boolean for BooleanFalse {}
    impl BooleanAtom for BooleanFalse {}

    impl BooleanFalse {
        pub const fn new() -> Self {
            let base = base!(is_negative = Some(true));
            BooleanFalse { base }
        }
    }

    impl From<BooleanFalse> for CalcursType {
        fn from(value: BooleanFalse) -> Self {
            CalcursType::BooleanFalse(value)
        }
    }

    impl From<BooleanFalse> for bool {
        fn from(_: BooleanFalse) -> bool {
            false
        }
    }

    #[calcurs_type(BooleanFunc)]
    #[derive(Debug, Clone)]
    pub struct And {
        left: Box<dyn Boolean>,
        right: Box<dyn Boolean>,
    }
    impl BooleanFunc for And {}
    impl Application for And {}

    impl PartialEq for And {
        fn eq(&self, other: &And) -> bool {
            &self.left == &other.left && &self.right == &other.right
        }
    }

    impl And {
        pub fn new(lhs: impl Boolean, rhs: impl Boolean) -> Self {
            let is_negative = match (lhs.base().is_negative, rhs.base().is_negative) {
                (Some(l), Some(r)) => Some(l && r),
                (_, _) => None,
            };

            And {
                base: base!(is_negative),
                left: DynBoolean::dyn_clone(&rhs),
                right: DynBoolean::dyn_clone(&lhs),
            }
        }
    }

    impl Basic for And {
        fn eval(&self) -> CalcursType {
            Boolean::eval_impl(self).as_calcrs_type()
        }
    }

    impl Boolean for And {
        fn eval_impl(&self) -> &'static dyn Boolean {
            let left = &self.left;
            let right = &self.right;

            let lhs = Boolean::eval_impl(left.as_ref());
            let rhs = Boolean::eval_impl(right.as_ref());

            match (lhs.base().is_negative, rhs.base().is_negative) {
                (Some(false), Some(false)) | (Some(true), Some(true)) => &TRUE,
                _ => &FALSE,
            }
        }
    }

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

pub const FALSE: BooleanFalse = BooleanFalse::new();
pub const TRUE: BooleanTrue = BooleanTrue::new();

#[cfg(test)]
mod test {

    use crate::*;

    #[test]
    fn boolean() {
        assert_eq!(And::new(TRUE, FALSE), TRUE & FALSE);
        assert_eq!(CalcursType::from(FALSE), (TRUE & FALSE).eval());
    }

    #[test]
    fn calcurs_traits() {
        assert!(!TRUE.base.is_negative.unwrap());
        assert!(FALSE.base.is_negative.unwrap());
        assert!(TRUE.base.is_atom);
        assert!(FALSE.base.is_atom);
        assert!(And::new_base().is_function);
        assert!(!And::new_base().is_number);
        assert!(And::new_base().is_boolean);
        assert!(!(TRUE & FALSE).base.is_negative.unwrap());
        assert!((FALSE & FALSE).base.is_negative.unwrap());
    }
}
