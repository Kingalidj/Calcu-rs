use crate::visit::ControlFlow;
use derive_more::{Display, From};
use std::any::TypeId;
use std::collections::BTreeSet;
use std::fmt;
use std::fmt::Formatter;
use std::marker::PhantomData;
use std::ops::Deref;
use visit::Visitor;

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

pub mod visit {
    use crate::{And, Basic, Boolean, BooleanAtom, Not, Or};

    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub enum ControlFlow {
        Return,
        Continue,
    }

    use ControlFlow as CF;

    impl ControlFlow {
        pub fn ret(b: bool) -> Self {
            match b {
                true => CF::Return,
                false => CF::Continue,
            }
        }

        pub fn cont(b: bool) -> Self {
            Self::ret(!b)
        }
    }

    macro_rules! early_return {
        ($e: expr) => {{
            match $e {
                CF::Return => return CF::Return,
                CF::Continue => CF::Continue,
            }
        }};
    }

    // first walk then visit

    pub fn walk_basic(v: &mut impl Visitor, b: &Basic) -> ControlFlow {
        early_return!(v.visit_basic(b));

        use Basic as B;
        match b {
            B::Boolean(b) => walk_boolean(v, b),
        }
    }

    pub fn walk_boolean(v: &mut impl Visitor, b: &Boolean) -> ControlFlow {
        early_return!(v.visit_boolean(b));

        use Boolean as B;
        match b {
            B::And(b) => walk_and(v, b),
            B::Or(b) => walk_or(v, b),
            B::Not(b) => walk_not(v, b),
            B::BooleanAtom(b) => walk_boolean_atom(v, b),
        }
    }

    pub fn walk_boolean_atom(v: &mut impl Visitor, b: &BooleanAtom) -> ControlFlow {
        early_return!(v.visit_boolean_atom(b));
        ControlFlow::Continue
    }

    pub fn walk_and(v: &mut impl Visitor, a: &And) -> ControlFlow {
        early_return!(v.visit_and(a));
        ControlFlow::Continue
    }

    pub fn walk_or(v: &mut impl Visitor, a: &Or) -> ControlFlow {
        early_return!(v.visit_or(a));
        ControlFlow::Continue
    }

    pub fn walk_not(v: &mut impl Visitor, a: &Not) -> ControlFlow {
        early_return!(v.visit_not(a));
        ControlFlow::Continue
    }

    pub trait Visitor: Sized {
        fn visit_basic(&mut self, _b: &Basic) -> ControlFlow {
            ControlFlow::Continue
        }
        fn visit_boolean_atom(&mut self, _b: &BooleanAtom) -> ControlFlow {
            ControlFlow::Continue
        }

        fn visit_boolean(&mut self, _b: &Boolean) -> ControlFlow {
            ControlFlow::Continue
        }

        fn visit_and(&mut self, _a: &And) -> ControlFlow {
            ControlFlow::Continue
        }

        fn visit_or(&mut self, _o: &Or) -> ControlFlow {
            ControlFlow::Continue
        }

        fn visit_not(&mut self, _n: &Not) -> ControlFlow {
            ControlFlow::Continue
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct IsVisitor<T> {
    val: bool,
    tag: PhantomData<T>,
}

impl<T: 'static> IsVisitor<T> {
    fn is(b: &Basic) -> bool {
        let mut vis = Self {
            val: false,
            tag: PhantomData,
        };

        visit::walk_basic(&mut vis, b);
        vis.val
    }
}

impl<T: 'static> Visitor for IsVisitor<T> {
    fn visit_basic(&mut self, _b: &Basic) -> ControlFlow {
        self.val = can_cast::<Basic, T>();
        ControlFlow::ret(self.val)
    }

    fn visit_boolean_atom(&mut self, _b: &BooleanAtom) -> ControlFlow {
        self.val = can_cast::<BooleanAtom, T>();
        ControlFlow::ret(self.val)
    }

    fn visit_boolean(&mut self, _b: &Boolean) -> ControlFlow {
        self.val = can_cast::<Boolean, T>();
        ControlFlow::ret(self.val)
    }

    fn visit_and(&mut self, _a: &And) -> ControlFlow {
        self.val = can_cast::<And, T>();
        ControlFlow::ret(self.val)
    }

    fn visit_or(&mut self, _o: &Or) -> ControlFlow {
        self.val = can_cast::<Or, T>();
        ControlFlow::ret(self.val)
    }

    fn visit_not(&mut self, _n: &Not) -> ControlFlow {
        self.val = can_cast::<Not, T>();
        ControlFlow::ret(self.val)
    }
}

#[derive(Debug, Clone, From)]
pub enum Basic {
    Boolean(Boolean),
}

impl Basic {
    pub fn is<T: 'static>(&self) -> bool {
        IsVisitor::<T>::is(self)
    }
}

#[derive(Debug, Clone, From, Display)]
pub enum Boolean {
    And(And),
    Or(Or),
    Not(Not),
    BooleanAtom(BooleanAtom),
}

impl From<bool> for Boolean {
    fn from(value: bool) -> Self {
        Boolean::BooleanAtom(value.into())
    }
}

#[derive(Debug, Clone, Copy, Default, Display)]
pub struct BooleanAtom {
    val: bool,
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

#[derive(Debug, Clone, Default)]
pub struct And {
    args: Vec<Boolean>,
}

impl fmt::Display for And {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let args = &self.args;

        if args.is_empty() {
            return Ok(());
        }

        let mut iter = args.iter();

        write!(f, "(")?;

        if let Some(b) = iter.next() {
            write!(f, "{b}")?;
        }

        while let Some(b) = iter.next() {
            write!(f, " ∧ {b}")?;
        }

        write!(f, ")")
    }
}

impl From<And> for Basic {
    fn from(b: And) -> Self {
        Basic::Boolean(b.into())
    }
}

#[derive(Debug, Clone, Display)]
pub struct Or {}

impl From<Or> for Basic {
    fn from(b: Or) -> Self {
        Basic::Boolean(b.into())
    }
}

#[derive(Debug, Clone, Display)]
pub struct Not {}

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

    #[test]
    fn test() {
        let mut a: And = And::default();
        a.args.push(false.into());
        a.args.push(true.into());
        a.args.push(false.into());
        a.args.push(true.into());

        println!("{a}");
    }
}
