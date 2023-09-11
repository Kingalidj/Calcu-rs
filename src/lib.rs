//! simple symbolic algebra system in rust

use core::fmt::Debug;
use paste::paste;
use std::ops;

#[derive(Debug, Clone, PartialEq)]
pub enum CalcursType {
    BooleanTrue(BooleanTrue),
    BooleanFalse(BooleanFalse),
    And(Box<And>),
    Symbol(Symbol),
}

macro_rules! for_each_type {
    ($e: ident: $v: ident => $func: tt) => {{
        use CalcursType::*;
        match $e {
            BooleanTrue($v) => $func,
            BooleanFalse($v) => $func,
            And($v) => {
                let $v = $v.as_ref();
                $func
            }
            Symbol($v) => $func,
        }
    }};
}

macro_rules! auto_ref {
    ($expr: expr) => {
        (&$expr)
    };
}

macro_rules! impl_getter {
    ($fn_name: ident: $impl_fn: ident -> $ty: ty) => {
        pub fn $fn_name(&self) -> $ty {
            for_each_type!(self: v => { auto_ref!(*v).$impl_fn() })
        }
    }
}

impl CalcursType {
    impl_getter!(is_atom: _is_atom -> bool);
    impl_getter!(is_boolean: _is_boolean -> bool);
    impl_getter!(is_function: _is_function -> bool);
    impl_getter!(is_scalar: _is_scalar -> bool);
}

macro_rules! specialized_trait {
    ([$basic: ident]: $fn_name: ident -> $ret_ty: ty { $special: ident: $spec_ret: expr }) => {
        paste! {
            pub trait [<__Default $fn_name:camel>] {
                #[inline]
                fn $fn_name(&self) -> $ret_ty {
                    Default::default()
                }
            }

            pub trait [<__ $fn_name:camel>] {
                #[inline]
                fn $fn_name(&self) -> $ret_ty {
                    $spec_ret
                }
            }

            impl<T: $basic> [<__Default $fn_name:camel>] for &T {}
            impl<T: $special> [<__ $fn_name:camel >] for T {}
        }
    };
}

specialized_trait!([Basic]: _is_atom -> bool { Atom: true });
specialized_trait!([Basic]: _is_boolean -> bool { Boolean: true });
specialized_trait!([Basic]: _is_function -> bool { Application: true });
specialized_trait!([Basic]: _is_scalar -> bool { Expr: true });

pub trait Basic: Debug + Into<CalcursType> + Clone + PartialEq {
    fn typ(self) -> CalcursType {
        self.into()
    }
}

pub trait Atom: Basic {}
pub trait Expr: Basic {}

pub trait Boolean: Basic {
    fn bool_val(&self) -> Option<bool>;
}
pub trait Application: Basic {}

pub trait AtomicExpr: Atom + Expr {}

pub trait BooleanAtom: Atom + Boolean {}

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
    pub const fn new(name: &'static str) -> Self {
        Self { name }
    }
}

impl From<Symbol> for CalcursType {
    fn from(value: Symbol) -> CalcursType {
        CalcursType::Symbol(value)
    }
}

#[derive(Debug, Clone, Default, Copy, PartialEq)]
pub struct BooleanTrue;

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
    lhs: CalcursType,
    rhs: CalcursType,
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
    pub fn new<S: Boolean, T: Boolean>(lhs: S, rhs: T) -> And {
        let value = AndValue(lhs.bool_val()).and(AndValue(rhs.bool_val()));
        And {
            lhs: lhs.into(),
            rhs: rhs.into(),
            value,
        }
    }

    pub const fn const_new(lhs: CalcursType, rhs: CalcursType, value: Option<bool>) -> And {
        And {
            lhs,
            rhs,
            value: AndValue(value),
        }
    }
}

impl From<And> for CalcursType {
    fn from(value: And) -> Self {
        CalcursType::And(value.into())
    }
}

impl<T: Boolean> ops::BitAnd<T> for BooleanTrue {
    type Output = And;

    fn bitand(self, rhs: T) -> Self::Output {
        And::new(self, rhs)
    }
}

impl<T: Boolean> ops::BitAnd<T> for BooleanFalse {
    type Output = And;

    fn bitand(self, rhs: T) -> Self::Output {
        And::new(self, rhs)
    }
}

impl<T: Boolean> ops::BitAnd<T> for And {
    type Output = And;

    fn bitand(self, rhs: T) -> Self::Output {
        And::new(self, rhs)
    }
}

impl<T: Boolean> ops::BitAnd<T> for Symbol {
    type Output = And;

    fn bitand(self, rhs: T) -> Self::Output {
        And::new(self, rhs)
    }
}

pub const FALSE: BooleanFalse = BooleanFalse;
pub const TRUE: BooleanTrue = BooleanTrue;

#[cfg(test)]
mod test {

    use crate::*;

    #[test]
    fn boolean() {
        assert_eq!(And::new(TRUE, FALSE), TRUE & FALSE);
    }

    const TRUE_TYP: CalcursType = CalcursType::BooleanTrue(TRUE);
    const FALSE_TYP: CalcursType = CalcursType::BooleanFalse(FALSE);

    const X: Symbol = Symbol::new("x");
    const X_TYP: CalcursType = CalcursType::Symbol(Symbol::new("x"));

    const TRUE_AND_TRUE: And = And::const_new(TRUE_TYP, TRUE_TYP, Some(true));
    const TRUE_AND_FALSE: And = And::const_new(TRUE_TYP, FALSE_TYP, Some(false));
    const TRUE_AND_X: And = And::const_new(TRUE_TYP, X_TYP, None);

    #[test]
    fn is_atom() {
        assert!(TRUE.typ().is_atom());
        assert!(FALSE.typ().is_atom());
        assert!(X.typ().is_atom());
        assert!(!TRUE_AND_TRUE.typ().is_atom());
        assert!(!TRUE_AND_FALSE.typ().is_atom());
        assert!(!TRUE_AND_X.typ().is_atom());
    }

    #[test]
    fn is_boolean() {
        assert!(TRUE.typ().is_boolean());
        assert!(FALSE.typ().is_boolean());
        assert!(X.typ().is_boolean());
        assert!(TRUE_AND_TRUE.typ().is_boolean());
        assert!(TRUE_AND_FALSE.typ().is_boolean());
        assert!(TRUE_AND_X.typ().is_boolean());
    }

    #[test]
    fn is_function() {
        assert!(!TRUE.typ().is_function());
        assert!(!FALSE.typ().is_function());
        assert!(!X.typ().is_function());
        assert!(TRUE_AND_TRUE.typ().is_function());
        assert!(TRUE_AND_FALSE.typ().is_function());
        assert!(TRUE_AND_X.typ().is_function());
    }

    #[test]
    fn is_scalar() {
        assert!(!TRUE.typ().is_scalar());
        assert!(!FALSE.typ().is_scalar());
        assert!(X.typ().is_scalar());
        assert!(!TRUE_AND_TRUE.typ().is_scalar());
        assert!(!TRUE_AND_FALSE.typ().is_scalar());
        assert!(!TRUE_AND_X.typ().is_scalar());
    }

    #[test]
    fn bool_logic() {
        assert!(TRUE.bool_val().unwrap());
        assert!(!FALSE.bool_val().unwrap());
        assert!(X.bool_val().is_none());
        assert!(TRUE_AND_TRUE.bool_val().unwrap());
        assert!(!TRUE_AND_FALSE.bool_val().unwrap());
        assert!(TRUE_AND_X.bool_val().is_none());
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
