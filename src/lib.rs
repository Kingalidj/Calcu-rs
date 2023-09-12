//! simple symbolic algebra system in rust

use calcurs_macros::dyn_trait;
use core::fmt::Debug;
use paste::paste;
use std::{fmt::Display, ops};

macro_rules! _specialized_trait {
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

#[derive(Clone)]
pub enum CalcursType {
    BooleanTrue(BooleanTrue),
    BooleanFalse(BooleanFalse),
    And(And),
    Symbol(Symbol),
}

macro_rules! wrap {
    ($type: ident: $val: expr) => {
        paste! {
            CalcursType::$type($val)
        }
    };
}

macro_rules! for_each_type {
    (match $e: ident |$v: ident| $func: tt) => {{
        use CalcursType::*;
        match $e {
            BooleanTrue($v) => $func,
            BooleanFalse($v) => $func,
            And($v) => $func,
            Symbol($v) => $func,
        }
    }};
}

impl Debug for CalcursType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for_each_type!(match self |v| { write!(f, "{:?}", v)})
    }
}

impl Display for CalcursType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for_each_type!(match self |v| { write!(f, "{}", v)})
    }
}

impl CalcursType {
    pub fn get_boolean(&self) -> Option<&dyn Boolean> {
        use CalcursType::*;
        match self {
            BooleanTrue(v) => Some(v),
            BooleanFalse(v) => Some(v),
            And(v) => Some(v),
            Symbol(v) => Some(v),
        }
    }

    pub fn is_boolean(&self) -> bool {
        self.get_boolean().is_some()
    }

    pub fn subs<SYM: Into<Symbol>>(&self, sym: SYM, value: CalcursType) -> CalcursType {
        for_each_type!(match self |v| { v.subs(sym.into(), value).unwrap_or(self.clone()) })
    }
}

impl<'a> From<&'a CalcursType> for Option<&'a dyn Boolean> {
    fn from(value: &'a CalcursType) -> Self {
        value.get_boolean()
    }
}

impl PartialEq for CalcursType {
    fn eq(&self, other: &Self) -> bool {
        if let (Some(b1), Some(b2)) = (self.get_boolean(), other.get_boolean()) {
            if let (Some(v1), Some(v2)) = (b1.bool_val(), b2.bool_val()) {
                return v1 == v2;
            }
        }

        false
    }
}

#[dyn_trait]
pub trait Substitude {
    fn subs(&self, _: Symbol, _: CalcursType) -> Option<CalcursType> {
        None
    }
}

#[dyn_trait]
pub trait Basic: Display + Debug + Substitude {
    fn as_type(&self) -> CalcursType;
}

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
    ident: &'static str,
}

impl Basic for Symbol {
    fn as_type(&self) -> CalcursType {
        (*self).into()
    }
}
impl Atom for Symbol {}
impl Expr for Symbol {}
impl AtomicExpr for Symbol {}
impl Boolean for Symbol {
    fn bool_val(&self) -> Option<bool> {
        None
    }
}

impl Display for Symbol {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.ident)
    }
}

impl Substitude for Symbol {
    fn subs(&self, sym: Symbol, value: CalcursType) -> Option<CalcursType> {
        match sym == *self {
            true => Some(value),
            false => None,
        }
    }
}

impl Symbol {
    pub const fn new(name: &'static str) -> Self {
        Self { ident: name }
    }

    pub const fn typ(name: &'static str) -> CalcursType {
        wrap!(Symbol: Self::new(name))
    }
}

impl From<&'static str> for Symbol {
    fn from(value: &'static str) -> Self {
        Symbol::new(value)
    }
}

impl From<Symbol> for CalcursType {
    fn from(value: Symbol) -> CalcursType {
        wrap!(Symbol: value)
    }
}

#[derive(Debug, Clone, Default, Copy, PartialEq)]
pub struct BooleanTrue;

impl BooleanTrue {
    pub const fn to_typ(self) -> CalcursType {
        wrap!(BooleanTrue: self)
    }

    pub const fn typ() -> CalcursType {
        Self {}.to_typ()
    }
}

impl Basic for BooleanTrue {
    fn as_type(&self) -> CalcursType {
        (*self).into()
    }
}

impl Atom for BooleanTrue {}
impl Boolean for BooleanTrue {
    fn bool_val(&self) -> Option<bool> {
        Some(true)
    }
}

impl Substitude for BooleanTrue {}

impl Display for BooleanTrue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "true")
    }
}

impl From<BooleanTrue> for CalcursType {
    fn from(value: BooleanTrue) -> Self {
        value.to_typ()
    }
}

#[derive(Debug, Clone, Default, Copy, PartialEq)]
pub struct BooleanFalse;

impl BooleanFalse {
    pub const fn typ() -> CalcursType {
        wrap!(BooleanFalse: Self {})
    }
}

impl Basic for BooleanFalse {
    fn as_type(&self) -> CalcursType {
        (*self).into()
    }
}

impl Atom for BooleanFalse {}
impl Boolean for BooleanFalse {
    fn bool_val(&self) -> Option<bool> {
        Some(false)
    }
}

impl Substitude for BooleanFalse {}

impl Display for BooleanFalse {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "false")
    }
}

impl From<BooleanFalse> for CalcursType {
    fn from(value: BooleanFalse) -> Self {
        wrap!(BooleanFalse: value)
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

impl And {
    pub fn new(lhs: Box<dyn Boolean>, rhs: Box<dyn Boolean>) -> Self {
        let v1 = AndValue(lhs.bool_val());
        let v2 = AndValue(rhs.bool_val());
        let value = v1.and(v2);
        And { lhs, rhs, value }
    }

    pub fn typ(lhs: Box<dyn Boolean>, rhs: Box<dyn Boolean>) -> CalcursType {
        wrap!(And: Self::new(lhs, rhs))
    }
}

impl Basic for And {
    fn as_type(&self) -> CalcursType {
        (*self).clone().into()
    }
}
impl Application for And {}
impl BooleanFunc for And {}
impl Boolean for And {
    fn bool_val(&self) -> Option<bool> {
        self.value.0
    }
}

impl Substitude for And {
    fn subs(&self, sym: Symbol, value: CalcursType) -> Option<CalcursType> {
        if !value.is_boolean() {
            return None;
        }

        let lhs = self.lhs.subs(sym, value.clone());
        let rhs = self.rhs.subs(sym, value.clone());

        match (&lhs, &rhs) {
            (None, None) => return None,
            _ => (),
        }

        let binding = lhs.unwrap_or(self.lhs.as_type());
        let lhs = binding.get_boolean().expect("should be unreachable");

        let binding = rhs.unwrap_or(self.rhs.as_type());
        let rhs = binding.get_boolean().expect("should be unreachable");

        Some(And::typ(
            DynBoolean::box_clone(lhs),
            DynBoolean::box_clone(rhs),
        ))
    }
}

impl Display for And {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({} & {})", self.lhs, self.rhs)
    }
}

impl PartialEq for And {
    fn eq(&self, other: &And) -> bool {
        self.value == other.value
    }
}

impl From<And> for CalcursType {
    fn from(value: And) -> Self {
        wrap!(And: value)
    }
}

impl ops::BitAnd<CalcursType> for CalcursType {
    type Output = CalcursType;

    fn bitand(self, rhs: CalcursType) -> Self::Output {
        match (self.get_boolean(), rhs.get_boolean()) {
            (Some(b1), Some(b2)) => And::typ(DynBoolean::box_clone(b1), DynBoolean::box_clone(b2)),
            _ => todo!(),
        }
    }
}

pub const FALSE: CalcursType = BooleanFalse::typ();
pub const TRUE: CalcursType = BooleanTrue::typ();

#[cfg(test)]
mod test {

    use crate::*;

    macro_rules! parse {
        (true) => {
            TRUE
        };

        (false) => {
            FALSE
        };

        ($e: ident) => {
            Symbol::typ(stringify!($e))
        };

        ($e1: ident $(& $e2: ident)+) => {
            parse!($e1) $(& parse!($e2))+
        };
    }

    #[test]
    fn is_boolean() {
        assert!(parse!(true).is_boolean());
        assert!(parse!(false).is_boolean());
        assert!(parse!(x).is_boolean());
        assert!(parse!(x & y).is_boolean());
    }

    #[test]
    fn bool_logic() {
        assert_eq!(parse!(true & true), TRUE);
        assert_eq!(parse!(true & false), FALSE);
        assert_eq!(parse!(false & false), parse!(false & true));
        assert_ne!(parse!(true & x), TRUE);
        assert_ne!(parse!(x & true), FALSE);
        assert!(parse!(x & y & z)
            .get_boolean()
            .unwrap()
            .bool_val()
            .is_none());
    }

    #[test]
    fn substitude() {
        let expr = parse!(x & y);

        assert_eq!(expr.subs("x", TRUE).subs("y", TRUE), TRUE);
        assert_eq!(expr.subs("x", FALSE).subs("y", TRUE), FALSE);
        assert_eq!(expr.subs("x", TRUE).subs("y", FALSE), FALSE);
    }
}
