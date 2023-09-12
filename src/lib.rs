//! simple symbolic algebra system in rust

use calcurs_macros::dyn_trait;
use core::fmt::Debug;
use paste::paste;
use std::ops;

#[derive(Debug, Clone)]
pub enum CalcursType {
    BooleanTrue(BooleanTrue),
    BooleanFalse(BooleanFalse),
    And(And),
    Symbol(Symbol),
}

macro_rules! for_each_type {
    ($e: ident: $v: ident => $func: tt) => {{
        use CalcursType::*;
        match $e {
            BooleanTrue($v) => $func,
            BooleanFalse($v) => $func,
            And($v) => $func,
            Symbol($v) => $func,
        }
    }};
}

macro_rules! impl_getter {
    ($fn_name: ident: $impl_fn: ident -> $ty: ty) => {
        pub fn $fn_name(&self) -> $ty {
            for_each_type!(self: v => { auto_ref!(*v).$impl_fn() })
        }
    }
}

macro_rules! auto_ref {
    ($expr: expr) => {
        (&$expr)
    };
}

impl CalcursType {
    impl_getter!(get_atom: get_atom_impl -> Option<Box<dyn Atom>>);
    pub fn is_atom(&self) -> bool {
        self.get_atom().is_some()
    }

    impl_getter!(get_boolean: get_boolean_impl -> Option<Box<dyn Boolean>>);
    pub fn is_boolean(&self) -> bool {
        self.get_boolean().is_some()
    }

    impl_getter!(is_function: is_function_impl -> bool);

    impl_getter!(is_scalar: is_scalar_impl -> bool);
}

macro_rules! specialized_trait {
    ([$basic: ident => $special: ident]: $fn_name: ident (&$self: ident) -> $ret_ty: ty { $spec_ret: expr }) => {
        paste! {
            pub trait [<__Default $fn_name:camel>] {
                #[inline]
                fn $fn_name(&self) -> $ret_ty {
                    Default::default()
                }
            }

            pub trait [<__ $fn_name:camel>] {
                fn $fn_name(&self) -> $ret_ty;
            }

            impl<T: ?Sized + $basic> [<__Default $fn_name:camel>] for &T {}
            impl<T: ?Sized + $special> [<__ $fn_name:camel >] for T {
                #[inline]
                fn $fn_name(&$self) -> $ret_ty {
                    $spec_ret
                }
            }
        }
    };
}

specialized_trait!([Basic => Atom]: get_atom_impl(&self) -> Option<Box<dyn Atom>> { Some(DynAtom::box_clone(self)) });

specialized_trait!([Basic => Boolean]: get_boolean_impl(&self) -> Option<Box<dyn Boolean>> {  Some(DynBoolean::box_clone(self))  });

specialized_trait!([Basic => Application]: is_function_impl(&self) -> bool { true });

specialized_trait!([Basic => Expr]: is_scalar_impl(&self) -> bool { true });

impl PartialEq for CalcursType {
    fn eq(&self, other: &Self) -> bool {
        match (self.get_boolean(), other.get_boolean()) {
            (Some(b1), Some(b2)) => match (b1.bool_val(), b2.bool_val()) {
                (Some(v1), Some(v2)) => return v1 == v2,
                _ => return false,
            },
            _ => false,
        }
    }
}

#[dyn_trait]
pub trait Basic: Debug {}

#[dyn_trait]
pub trait Atom: Basic {}

#[dyn_trait]
pub trait Expr: Basic {}

#[dyn_trait]
pub trait Boolean: Basic {
    fn bool_val(&self) -> Option<bool>;
}

#[dyn_trait]
pub trait Application: Basic {}

#[dyn_trait]
pub trait AtomicExpr: Atom + Expr {}

#[dyn_trait]
pub trait BooleanAtom: Atom + Boolean {}

#[dyn_trait]
pub trait BooleanFunc: Boolean + Application {}

#[derive(Debug, Clone, Default, Copy, PartialEq)]
pub struct Symbol {
    name: &'static str,
}

impl Basic for Symbol {}
impl Atom for Symbol {}
impl Expr for Symbol {}
impl AtomicExpr for Symbol {}
impl Boolean for Symbol {
    fn bool_val(&self) -> Option<bool> {
        None
    }
}

impl Symbol {
    pub const fn new(name: &'static str) -> CalcursType {
        CalcursType::Symbol(Self { name })
    }
}

impl From<Symbol> for CalcursType {
    fn from(value: Symbol) -> CalcursType {
        CalcursType::Symbol(value)
    }
}

#[derive(Debug, Clone, Default, Copy, PartialEq)]
pub struct BooleanTrue;

impl BooleanTrue {
    pub const fn new() -> CalcursType {
        CalcursType::BooleanTrue(Self {})
    }
}

impl Basic for BooleanTrue {}
impl Atom for BooleanTrue {}
impl Boolean for BooleanTrue {
    fn bool_val(&self) -> Option<bool> {
        Some(true)
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

#[derive(Debug, Clone, Default, Copy, PartialEq)]
pub struct BooleanFalse;

impl BooleanFalse {
    pub const fn new() -> CalcursType {
        CalcursType::BooleanFalse(Self {})
    }
}

impl Basic for BooleanFalse {}
impl Atom for BooleanFalse {}
impl Boolean for BooleanFalse {
    fn bool_val(&self) -> Option<bool> {
        Some(false)
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

#[derive(Debug, Clone, Copy, PartialEq)]
struct AndValue(Option<bool>);

impl AndValue {
    fn and(self, rhs: AndValue) -> Self {
        let val = match (self.0, rhs.0) {
            (Some(v1), Some(v2)) => Some(v1 && v2),
            _ => None,
        };
        Self(val)
    }
}

#[derive(Debug, Clone)]
pub struct And {
    lhs: Box<dyn Boolean>,
    rhs: Box<dyn Boolean>,
    value: AndValue,
}

impl Basic for And {}
impl Application for And {}
impl BooleanFunc for And {}
impl Boolean for And {
    fn bool_val(&self) -> Option<bool> {
        self.value.0
    }
}

impl PartialEq for And {
    fn eq(&self, other: &And) -> bool {
        self.value == other.value
    }
}

impl And {
    pub fn new(lhs: Box<dyn Boolean>, rhs: Box<dyn Boolean>) -> CalcursType {
        let value = AndValue(lhs.bool_val()).and(AndValue(rhs.bool_val()));
        CalcursType::And(And { lhs, rhs, value })
    }
}

impl From<And> for CalcursType {
    fn from(value: And) -> Self {
        CalcursType::And(value.into())
    }
}

impl ops::BitAnd<CalcursType> for CalcursType {
    type Output = CalcursType;

    fn bitand(self, rhs: CalcursType) -> Self::Output {
        match (self.get_boolean(), rhs.get_boolean()) {
            (Some(b1), Some(b2)) => And::new(b1, b2),
            _ => todo!(),
        }
    }
}

pub const FALSE: CalcursType = BooleanFalse::new();
pub const TRUE: CalcursType = BooleanTrue::new();

#[cfg(test)]
mod test {

    use crate::*;

    #[test]
    fn boolean() {
        // assert_eq!(And::new(TRUE, FALSE), TRUE & FALSE);
    }

    const X: CalcursType = Symbol::new("x");

    #[test]
    fn is_atom() {
        let true_and_true = TRUE & TRUE;
        let true_and_false = TRUE & FALSE;
        let true_and_x = TRUE & X;

        assert!(TRUE.is_atom());
        assert!(FALSE.is_atom());
        assert!(X.is_atom());
        assert!(!true_and_true.is_atom());
        assert!(!true_and_false.is_atom());
        assert!(!true_and_x.is_atom());
    }

    #[test]
    fn is_boolean() {
        let true_and_true = TRUE & TRUE;
        let true_and_false = TRUE & FALSE;
        let true_and_x = TRUE & X;

        assert!(TRUE.is_boolean());
        assert!(FALSE.is_boolean());
        assert!(X.is_boolean());
        assert!(true_and_true.is_boolean());
        assert!(true_and_false.is_boolean());
        assert!(true_and_x.is_boolean());
    }

    #[test]
    fn is_function() {
        let true_and_true = TRUE & TRUE;
        let true_and_false = TRUE & FALSE;
        let true_and_x = TRUE & X;

        assert!(!TRUE.is_function());
        assert!(!FALSE.is_function());
        assert!(!X.is_function());
        assert!(true_and_true.is_function());
        assert!(true_and_false.is_function());
        assert!(true_and_x.is_function());
    }

    #[test]
    fn is_scalar() {
        let true_and_true = TRUE & TRUE;
        let true_and_false = TRUE & FALSE;
        let true_and_x = TRUE & X;

        assert!(!TRUE.is_scalar());
        assert!(!FALSE.is_scalar());
        assert!(X.is_scalar());
        assert!(!true_and_true.is_scalar());
        assert!(!true_and_false.is_scalar());
        assert!(!true_and_x.is_scalar());
    }

    #[test]
    fn bool_logic() {
        let true_and_true = TRUE & TRUE;
        let true_and_false = TRUE & FALSE;
        let false_and_false = FALSE & FALSE;
        let true_and_x = TRUE & X;

        assert_eq!(true_and_true, TRUE);
        assert_eq!(true_and_false, FALSE);
        assert_eq!(false_and_false, FALSE);
        assert_ne!(true_and_x, TRUE);
        assert_ne!(true_and_x, FALSE);
    }
}

// is_number: bool,
// is_atom: bool,
// is_symbol: bool,
// is_function: bool,
// is_add: bool,
// is_mul: bool,
// is_pow: bool,
// is_float: bool,
// is_rational: bool,
// is_integer: bool,
// is_numbersymbol: bool,
// is_order: bool,
// is_derivative: bool,
// is_piecewise: bool,
// is_poly: bool,
// is_algebraicnumber: bool,
// is_relational: bool,
// is_equality: bool,
// is_boolean: bool,
// is_not: bool,
// is_matrix: bool,
// is_vector: bool,
// is_point: bool,
// is_matadd: bool,
// is_scalar: bool,
// is_matmul: Option<bool>,
// is_real: Option<bool>,
// is_zero: Option<bool>,
// is_negative: Option<bool>,
// is_commutative: Option<bool>,
