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
    Symbol(Symbol),
}

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
        pub is_scalar: bool,
    }

    pub trait IsCalcursType {
        fn to_calcurs_type(&self) -> CalcursType;
    }

    impl<T> IsCalcursType for T
    where
        T: Basic + Clone + Into<CalcursType> + PartialEq + 'static,
    {
        fn to_calcurs_type(&self) -> CalcursType {
            self.clone().into()
        }
    }

    pub trait Simplify {
        fn simplify_obj(&self) -> Box<dyn Basic>;

        fn simplify(&self) -> CalcursType {
            self.simplify_obj().to_calcurs_type()
        }
    }

    trait DefaultSimplify {}
    impl<T: Basic + DefaultSimplify> Simplify for T {
        fn simplify_obj(&self) -> Box<dyn Basic> {
            self.to_basic()
        }
    }

    pub trait Substitute {
        fn subs(&self, name: &'static str, val: Box<dyn Basic>) -> Box<dyn Basic>;
    }

    trait DefaultSubstitute {}
    impl<T: Basic + DefaultSubstitute> Substitute for T {
        fn subs(&self, _: &'static str, _: Box<dyn Basic>) -> Box<dyn Basic> {
            self.to_basic()
        }
    }

    #[dyn_trait]
    #[calcurs_trait()]
    pub trait Basic: Debug + IsCalcursType + Inherited<Base> + Simplify + Substitute {
        fn to_basic(&self) -> Box<dyn Basic> {
            self.box_clone()
        }
    }

    #[dyn_trait]
    #[calcurs_trait(is_atom = true)]
    pub trait Atom: Basic {}

    #[dyn_trait]
    #[calcurs_trait()]
    pub trait Boolean: Basic {}

    #[dyn_trait]
    #[calcurs_trait(is_scalar = true)]
    pub trait Expr: Basic {}

    #[dyn_trait]
    #[calcurs_trait()]
    pub trait AtomicExpr: Atom + Expr {}

    #[dyn_trait]
    #[calcurs_trait(is_boolean = true)]
    pub trait BooleanAtom: Atom + Boolean {}

    #[dyn_trait]
    #[calcurs_trait(is_function = true)]
    pub trait Application: Basic {}

    #[dyn_trait]
    #[calcurs_trait(is_boolean = true)]
    pub trait BooleanFunc: Boolean + Application {}

    #[calcurs_type(AtomicExpr + Boolean)]
    #[derive(Debug, Clone, Default, Copy, PartialEq)]
    pub struct Symbol {
        name: &'static str,
    }

    impl DefaultSimplify for Symbol {}

    impl Substitute for Symbol {
        fn subs(&self, name: &'static str, val: Box<dyn Basic>) -> Box<dyn Basic> {
            if self.name == name {
                val
            } else {
                self.to_basic()
            }
        }
    }

    impl Symbol {
        pub fn new(name: &'static str) -> Self {
            let base = base!(is_symbol = true);
            Self { name, base }
        }
    }

    impl From<Symbol> for CalcursType {
        fn from(value: Symbol) -> CalcursType {
            CalcursType::Symbol(value)
        }
    }

    #[calcurs_type(BooleanAtom)]
    #[derive(Debug, Clone, Default, Copy, PartialEq)]
    pub struct BooleanTrue {}

    impl DefaultSubstitute for BooleanTrue {}
    impl DefaultSimplify for BooleanTrue {}

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

    impl DefaultSubstitute for BooleanFalse {}
    impl DefaultSimplify for BooleanFalse {}

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

    // logical And: e.g: A && B
    #[calcurs_type(BooleanFunc)]
    #[derive(Debug, Clone)]
    pub struct And {
        left: Box<dyn Basic>,
        right: Box<dyn Basic>,
    }

    impl Substitute for And {
        fn subs(&self, name: &'static str, val: Box<dyn Basic>) -> Box<dyn Basic> {
            let mut res = self.clone();
            res.left = self.left.subs(name, val.clone());
            res.right = self.right.subs(name, val.clone());
            res.to_basic()
        }
    }

    impl Simplify for And {
        fn simplify_obj(&self) -> Box<dyn Basic> {
            self.simplify_impl().to_basic()
        }
    }

    impl PartialEq for And {
        fn eq(&self, other: &And) -> bool {
            let lhs1 = self.left.base().is_negative;
            let rhs1 = self.right.base().is_negative;
            let lhs2 = other.left.base().is_negative;
            let rhs2 = other.right.base().is_negative;

            match ((lhs1, rhs1), (lhs2, rhs2)) {
                ((Some(l1), Some(r1)), (Some(l2), Some(r2))) => l1 == l2 && r1 == r2,
                _ => false,
            }
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
                left: rhs.to_basic(),
                right: lhs.to_basic(),
            }
        }

        fn simplify_impl(&self) -> &'static dyn Boolean {
            let lhs = self.left.simplify_obj();
            let rhs = self.right.simplify_obj();

            match (lhs.base().is_negative, rhs.base().is_negative) {
                (Some(false), Some(false)) | (Some(true), Some(true)) => &TRUE,
                _ => &FALSE,
            }
        }
    }
}

impl From<And> for CalcursType {
    fn from(value: And) -> Self {
        CalcursType::And(value)
    }
}

impl<U: Boolean> ops::BitAnd<U> for Symbol {
    type Output = And;

    fn bitand(self, rhs: U) -> Self::Output {
        And::new(self, rhs)
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
        assert_eq!(CalcursType::from(FALSE), (TRUE & FALSE).simplify());
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

        assert!(Symbol::new("test").base().is_atom);
        assert!(Symbol::new("test").base().is_scalar);
        assert!(Symbol::new("test").base().is_symbol);
    }
}
