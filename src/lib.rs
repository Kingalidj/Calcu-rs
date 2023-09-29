use derive_more::{Display, From};
use std::any::TypeId;
use std::fmt;
use std::fmt::Formatter;
use std::marker::PhantomData;
use std::ops::Deref;
use std::rc::Rc;
use visitor::{DownCastVisitor, IsVisitor};

mod visitor;

/*
    precedence order:

    Not
    And
    Or
*/

pub fn can_cast<T: 'static, U: 'static>() -> bool {
    TypeId::of::<T>() == TypeId::of::<U>()
}

pub fn dyn_cast<T: 'static, U: 'static>(t: T) -> Result<U, T> {
    if can_cast::<T, U>() {
        let ptr = std::mem::ManuallyDrop::new(t).deref() as *const T as *const U;
        Ok(unsafe { std::ptr::read(ptr) })
    } else {
        Err(t)
    }
}

pub trait CloneByVal: Clone {
    fn clone_val(&self) -> Self;
}

#[derive(Debug, Clone, Display, PartialEq, Eq)]
pub struct RCP<T> {
    ptr: Rc<T>,
}

impl<T: CloneByVal> CloneByVal for RCP<T> {
    fn clone_val(&self) -> Self {
        Self {
            ptr: Rc::new(self.ptr.clone_val()),
        }
    }
}

impl<T: CloneByVal> CloneByVal for Vec<T> {
    fn clone_val(&self) -> Self {
        self.iter().map(|x| x.clone_val()).collect()
    }
}

impl<T> AsRef<T> for RCP<T> {
    fn as_ref(&self) -> &T {
        self.ptr.as_ref()
    }
}

impl<T> Deref for RCP<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.ptr.as_ref()
    }
}

impl<T> RCP<T> {
    pub fn new(val: T) -> Self {
        Self { ptr: val.into() }
    }
}

pub enum MaybeRef<T: CloneByVal> {
    Owned(T),
    Ref(RCP<T>),
}

impl<T: CloneByVal> MaybeRef<T> {
    pub fn borrow(val: RCP<T>) -> Self {
        MaybeRef::Ref(val)
    }

    pub fn take(val: T) -> Self {
        MaybeRef::Owned(val)
    }

    pub fn to_owned(self) -> T {
        match self {
            MaybeRef::Owned(v) => v,
            MaybeRef::Ref(v) => v.as_ref().clone_val(),
        }
    }

    pub fn to_ref(self) -> RCP<T> {
        match self {
            MaybeRef::Owned(v) => RCP::new(v),
            MaybeRef::Ref(v) => v,
        }
    }
}

#[derive(Debug, Clone, From, Display)]
pub enum Basic {
    Boolean(Boolean),
    Symbol(Symbol),
}

impl CloneByVal for Basic {
    fn clone_val(&self) -> Self {
        use Basic as B;
        match self {
            B::Boolean(b) => b.clone_val().into(),
            B::Symbol(s) => s.clone_val().into(),
        }
    }
}

impl Basic {
    pub fn is<T: 'static>(&self) -> bool {
        IsVisitor::<T>::is(self)
    }

    pub fn downcast<T: 'static>(&self) -> Option<T> {
        DownCastVisitor::downcast(self)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Symbol {
    name: RCP<String>,
}

impl CloneByVal for Symbol {
    fn clone_val(&self) -> Self {
        self.clone()
    }
}

impl fmt::Display for Symbol {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name)
    }
}

impl Symbol {
    pub fn new(name: RCP<String>) -> Self {
        Self { name }
    }
}

#[derive(Debug, Clone, From, Display)]
pub enum Boolean {
    And(And),
    Or(Or),
    Not(Not),
    BooleanAtom(BooleanAtom),
}

impl CloneByVal for Boolean {
    fn clone_val(&self) -> Self {
        use Boolean as B;
        match self {
            B::And(b) => b.clone_val().into(),
            B::Or(b) => b.clone_val().into(),
            B::Not(b) => b.clone_val().into(),
            B::BooleanAtom(b) => b.clone_val().into(),
        }
    }
}

impl From<bool> for Boolean {
    fn from(value: bool) -> Self {
        Boolean::BooleanAtom(value.into())
    }
}

#[derive(Debug, Clone, Default)]
pub struct AssociativeSet<OP: BinaryOperator> {
    set: Vec<Boolean>,
    tag: PhantomData<OP>,
}

impl<OP: BinaryOperator> fmt::Display for AssociativeSet<OP> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let args = &self.set;

        if args.is_empty() {
            return Ok(());
        }

        let mut iter = args.iter();

        write!(f, "(")?;

        if let Some(b) = iter.next() {
            write!(f, "{b}")?;
        }

        while let Some(b) = iter.next() {
            write!(f, " {} {b}", OP::REPR)?;
        }

        write!(f, ")")
    }
}

impl<OP: BinaryOperator> AssociativeSet<OP> {
    pub fn push(&mut self, b: Boolean) {
        self.set.push(b)
    }
}

#[derive(Debug, Clone, Copy, Default, Display)]
pub struct BooleanAtom {
    val: bool,
}

impl CloneByVal for BooleanAtom {
    fn clone_val(&self) -> Self {
        self.clone()
    }
}

impl From<bool> for BooleanAtom {
    fn from(val: bool) -> Self {
        Self { val }
    }
}

impl From<BooleanAtom> for Basic {
    fn from(b: BooleanAtom) -> Self {
        Basic::Boolean(b.into())
    }
}

pub trait BinaryOperator {
    const REPR: &'static str;
}

/// Canonical [And] is implemented with a [Boolean] coeff and a set of [CanonicalOr] of the form [{c1, v1}, {c2, v2}, ...] \
/// Example: \
/// coeff ∧ (c1 ∨ v1) ∧ (c2 ∨ v2) ∧ ... \
/// where the c's are all booleans and the v's can be any symbolic expression
#[derive(Debug, Clone, Display)]
#[display(fmt = "AND")]
pub struct And {
    /// extract all booleans to coeff
    coeff: RCP<Boolean>,
    /// create sum of product form
    args: RCP<Vec<Or>>,
}

impl CloneByVal for And {
    fn clone_val(&self) -> Self {
        Self {
            coeff: self.coeff.clone_val(),
            args: self.args.clone_val(),
        }
    }
}

pub fn and(lhs: RCP<Basic>, rhs: RCP<Basic>) -> And {
    if lhs.is::<And>() && rhs.is::<And>() {
        let lhs = lhs.downcast::<And>().unwrap();
        let rhs = rhs.downcast::<And>().unwrap();
    }

    todo!()
}

// pub struct And {
//     args: Vec<Basic>,
// }

impl BinaryOperator for And {
    const REPR: &'static str = "∧";
}

impl From<And> for Basic {
    fn from(b: And) -> Self {
        Basic::Boolean(b.into())
    }
}

/// Canonical [Or] is implemented with a [Boolean] coeff and a [Basic] value \
/// Example: \
/// coeff ∨ b
#[derive(Debug, Clone, Display)]
#[display(fmt = "OR")]
pub struct Or {
    /// extract all booleans to coeff
    coeff: RCP<Boolean>,
    /// rest
    value: RCP<Basic>,
}

impl CloneByVal for Or {
    fn clone_val(&self) -> Self {
        Self {
            coeff: self.coeff.clone_val(),
            value: self.value.clone_val(),
        }
    }
}

impl BinaryOperator for Or {
    const REPR: &'static str = "∨";
}

impl From<Or> for Basic {
    fn from(b: Or) -> Self {
        Basic::Boolean(b.into())
    }
}

#[derive(Debug, Clone)]
pub struct Not {
    val: RCP<Basic>,
}

impl CloneByVal for Not {
    fn clone_val(&self) -> Self {
        Self {
            val: self.val.clone_val(),
        }
    }
}

impl fmt::Display for Not {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "¬{}", self.val)
    }
}

impl From<Not> for Basic {
    fn from(b: Not) -> Self {
        Basic::Boolean(b.into())
    }
}

#[cfg(test)]
mod test {

    use crate::*;

    #[test]
    fn is() {
        let b: Basic = BooleanAtom::from(false).into();

        assert!(b.is::<BooleanAtom>());
        assert!(b.is::<Boolean>());
        assert!(b.is::<Basic>());

        assert!(!b.is::<Not>());
        assert!(!b.is::<And>());
        assert!(!b.is::<Or>());
    }
}
