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
    fn value(&self) -> Option<bool>;
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
    fn value(&self) -> Option<bool> {
        None
    }
}

impl Symbol {
    pub fn new(name: &'static str) -> Self {
        Self { name }
    }
}

impl From<Symbol> for CalcursType {
    fn from(value: Symbol) -> CalcursType {
        CalcursType::Symbol(value)
    }
}

#[derive(Debug, Clone, Default, Copy, PartialEq)]
pub struct BooleanTrue {}

impl Basic for BooleanTrue {}
impl Atom for BooleanTrue {}
impl Boolean for BooleanTrue {
    fn value(&self) -> Option<bool> {
        Some(true)
    }
}

impl BooleanTrue {
    pub const fn new() -> Self {
        BooleanTrue {}
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
pub struct BooleanFalse {}

impl Basic for BooleanFalse {}
impl Atom for BooleanFalse {}
impl Boolean for BooleanFalse {
    fn value(&self) -> Option<bool> {
        Some(false)
    }
}

impl BooleanFalse {
    pub const fn new() -> Self {
        BooleanFalse {}
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

impl ops::BitAnd<AndValue> for AndValue {
    type Output = AndValue;

    fn bitand(self, rhs: AndValue) -> Self::Output {
        let val = match (self.0, rhs.0) {
            (Some(v1), Some(v2)) => Some(v1 && v2),
            _ => None,
        };
        Self(val)
    }
}

#[derive(Debug, Clone)]
pub struct And {
    left: CalcursType,
    right: CalcursType,
    value: AndValue,
}

impl Basic for And {}
impl Application for And {}
impl BooleanFunc for And {}
impl Boolean for And {
    fn value(&self) -> Option<bool> {
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
        let value = AndValue(lhs.value()) & AndValue(rhs.value());
        And {
            left: lhs.into(),
            right: rhs.into(),
            value,
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

pub const FALSE: BooleanFalse = BooleanFalse::new();
pub const TRUE: BooleanTrue = BooleanTrue::new();

#[cfg(test)]
mod test {

    use crate::*;

    #[test]
    fn boolean() {
        assert_eq!(And::new(TRUE, FALSE), TRUE & FALSE);
    }

    #[test]
    fn is_atom() {
        let x = Symbol::new("x");
        let true_and_true = (TRUE & TRUE).typ();
        let true_and_false = (TRUE & FALSE).typ();
        let true_and_x = (TRUE & x.clone()).typ();

        assert!(TRUE.typ().is_atom());
        assert!(FALSE.typ().is_atom());
        assert!(x.typ().is_atom());
        assert!(!true_and_true.is_atom());
        assert!(!true_and_false.is_atom());
        assert!(!true_and_x.is_atom());
    }

    #[test]
    fn is_boolean() {
        let x = Symbol::new("x");
        let true_and_true = (TRUE & TRUE).typ();
        let true_and_false = (TRUE & FALSE).typ();
        let true_and_x = (TRUE & x.clone()).typ();

        assert!(TRUE.typ().is_boolean());
        assert!(FALSE.typ().is_boolean());
        assert!(x.typ().is_boolean());
        assert!(true_and_true.is_boolean());
        assert!(true_and_false.is_boolean());
        assert!(true_and_x.is_boolean());
    }

    #[test]
    fn is_function() {
        let x = Symbol::new("x");
        let true_and_true = (TRUE & TRUE).typ();
        let true_and_false = (TRUE & FALSE).typ();
        let true_and_x = (TRUE & x.clone()).typ();

        assert!(!TRUE.typ().is_function());
        assert!(!FALSE.typ().is_function());
        assert!(!x.typ().is_function());
        assert!(true_and_true.is_function());
        assert!(true_and_false.is_function());
        assert!(true_and_x.is_function());
    }

    #[test]
    fn is_scalar() {
        let x = Symbol::new("x");
        let true_and_true = (TRUE & TRUE).typ();
        let true_and_false = (TRUE & FALSE).typ();
        let true_and_x = (TRUE & x.clone()).typ();

        assert!(!TRUE.typ().is_scalar());
        assert!(!FALSE.typ().is_scalar());
        assert!(x.typ().is_scalar());
        assert!(!true_and_true.is_scalar());
        assert!(!true_and_false.is_scalar());
        assert!(!true_and_x.is_scalar());
    }

    #[test]
    fn bool_logic() {
        let x = Symbol::new("x");
        assert!((TRUE & TRUE).value().unwrap());
        assert!(!(TRUE & FALSE).value().unwrap());
        assert!(!(FALSE & TRUE).value().unwrap());
        assert!(!(FALSE & FALSE).value().unwrap());
        assert!((FALSE & x).value().is_none());
        assert!((x & TRUE).value().is_none());
        assert!((x & x).value().is_none());

        assert!(((TRUE & TRUE) & TRUE).value().unwrap());
        assert!(!((TRUE & TRUE) & FALSE).value().unwrap());
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
