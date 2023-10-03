use std::collections::HashSet;

#[inline(always)]
pub const fn is_same<T: CalcursType, U: CalcursType>() -> bool {
    T::ID as u32 == U::ID as u32
}

#[inline(always)]
pub const fn cast_ref<'a, T: CalcursType, U: CalcursType>(r#ref: &'a T) -> Option<&'a U> {
    if is_same::<T, U>() {
        let ptr = r#ref as *const T as *const U;
        let cast = unsafe { &*ptr };
        Some(cast)
    } else {
        None
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(u32)]
pub enum TypeID {
    Basic,
    Symbol,

    Boolean,
    BooleanAtom,
    And,
    Or,
    Not,

    #[default]
    Dummy,
}

pub trait CalcursType: Clone {
    const ID: TypeID;

    fn to_basic(self) -> Basic;
}

macro_rules! early_ret {
    ($e: expr) => {
        if let Some(ret) = $e {
            return Some(ret);
        }
    };
}

// should be completely optimized away => becomes same as just pattern matching
macro_rules! get_ref_impl {
    () => {
        #[inline(always)]
        pub const fn get_ref<T: CalcursType>(&self) -> Option<&T> {
            cast_ref::<Self, T>(self)
        }
    };

    ($($x: ident)+) => {
        #[inline(always)]
        pub const fn get_ref<T: CalcursType>(&self) -> Option<&T> {
            early_ret!(cast_ref::<Self, T>(self));
            $( early_ret!(self.$x.get_ref::<T>()); )+
            None
        }
    }
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct Basic {
    pub kind: BasicKind,
}

impl Basic {
    get_ref_impl!(kind);

    pub const fn is<T: CalcursType>(&self) -> bool {
        self.get_ref::<T>().is_some()
    }
}

impl CalcursType for Basic {
    const ID: TypeID = TypeID::Basic;

    fn to_basic(self) -> Self {
        self
    }
}

impl From<BasicKind> for Basic {
    fn from(value: BasicKind) -> Self {
        Basic { kind: value }
    }
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub enum BasicKind {
    Symbol(Symbol),
    Boolean(Boolean),

    Dummy,
}

impl BasicKind {
    #[inline(always)]
    pub const fn get_ref<T: CalcursType>(&self) -> Option<&T> {
        use BasicKind as BK;
        match self {
            BK::Symbol(b) => b.get_ref::<T>(),
            BK::Boolean(b) => b.get_ref::<T>(),
            BK::Dummy => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Symbol {
    pub name: String,
}

impl Symbol {
    pub fn new<I: Into<String>>(name: I) -> Self {
        Self { name: name.into() }
    }
}

impl CalcursType for Symbol {
    const ID: TypeID = TypeID::Symbol;

    fn to_basic(self) -> Basic {
        BasicKind::Symbol(self).into()
    }
}

impl Symbol {
    get_ref_impl!();
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct Boolean {
    pub kind: BooleanKind,
}

impl From<BooleanKind> for Boolean {
    fn from(value: BooleanKind) -> Self {
        Boolean { kind: value }
    }
}

impl CalcursType for Boolean {
    const ID: TypeID = TypeID::Boolean;

    fn to_basic(self) -> Basic {
        BasicKind::Boolean(self).into()
    }
}

impl Boolean {
    get_ref_impl!(kind);

    pub fn to_basic(self) -> Basic {
        BasicKind::Boolean(self).into()
    }
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub enum BooleanKind {
    Atom(BooleanAtom),

    And(And),
    Or(Or),
    Not(Not),
}

impl BooleanKind {
    #[inline(always)]
    pub const fn get_ref<T: CalcursType>(&self) -> Option<&T> {
        use BooleanKind as BK;
        match self {
            BK::Atom(b) => b.get_ref::<T>(),
            BK::And(b) => b.get_ref::<T>(),
            BK::Or(b) => b.get_ref::<T>(),
            BK::Not(b) => b.get_ref::<T>(),
        }
    }

    pub fn to_basic(self) -> Basic {
        BasicKind::Boolean(self.into()).into()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BooleanAtom {
    pub val: bool,
}

impl From<BooleanAtom> for bool {
    fn from(value: BooleanAtom) -> Self {
        value.val
    }
}

impl From<bool> for BooleanAtom {
    fn from(value: bool) -> Self {
        BooleanAtom { val: value }
    }
}

impl BooleanAtom {
    get_ref_impl!();
}

impl CalcursType for BooleanAtom {
    const ID: TypeID = TypeID::BooleanAtom;

    fn to_basic(self) -> Basic {
        BooleanKind::Atom(self).to_basic()
    }
}

/// Logical And. Implemented with a list of [Basic]
///
/// The only simplifications performed by the default [And] are: \
/// (a & b) & (c & d) => a & b & c & d (flatten And) \
/// a & b & c & a => a & b & c,  \
/// a & b & c & false => false,  \
/// a & b & c & true => a & b & c,
///
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct And {
    args: HashSet<Basic>,
}

/// Helper enum for extracting a subset of [Basic] often used in boolean Algebra, e.g [And], [Or],
/// etc...
#[derive(Debug)]
enum BooleanArg {
    True,
    False,
    And(And),
    Or(Or),
    Not(Not),
    Symbol(Symbol),
    //Other(Basic),
    Dummy,
}

impl BooleanArg {
    fn from_basic(b: Basic) -> Self {
        use BasicKind as B;
        use BooleanKind as BK;

        let b = match b.kind {
            B::Symbol(s) => return Self::Symbol(s),
            B::Boolean(b) => b,
            B::Dummy => return Self::Dummy,
        };

        match b.kind {
            BK::Atom(BooleanAtom { val }) => match val {
                true => Self::True,
                false => Self::False,
            },
            BK::And(a) => Self::And(a),
            BK::Or(o) => Self::Or(o),
            BK::Not(n) => Self::Not(n),
        }
    }

    fn to_basic(self) -> Basic {
        use BooleanArg as BA;
        use BooleanKind as BK;
        match self {
            BA::True => TRUE,
            BA::False => FALSE,
            BA::And(and) => BK::And(and).to_basic(),
            BA::Or(or) => BK::Or(or).to_basic(),
            BA::Not(not) => BK::Not(not).to_basic(),
            BA::Symbol(sym) => BasicKind::Symbol(sym).into(),
            //BA::Other(basic) => basic,
            BA::Dummy => BasicKind::Dummy.into(),
        }
    }
}

impl std::hash::Hash for And {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        "HashSet".hash(state);
        for a in &self.args {
            a.hash(state)
        }
    }
}

impl And {
    get_ref_impl!();

    pub fn and<S: CalcursType, T: CalcursType>(b1: S, b2: T) -> Boolean {
        let res = Self::from_arg(b1.to_basic());

        if let BooleanKind::And(and) = res.kind {
            and.insert_arg(b2.to_basic())
        } else {
            return res;
        }
    }

    fn uneval_insert(mut self, b: Basic) -> Self {
        self.args.insert(b);
        self
    }

    fn insert_arg(self, b: Basic) -> Boolean {
        self.insert_bool_arg(BooleanArg::from_basic(b))
    }

    fn from_arg(arg: Basic) -> Boolean {
        Self {
            args: HashSet::new(),
        }
        .insert_arg(arg)
    }

    fn extend<I: IntoIterator<Item = Basic>>(mut self, args: I) -> Self {
        self.args.extend(args);
        self
    }

    fn insert_bool_arg(self, b: BooleanArg) -> Boolean {
        use BooleanArg as BA;

        BooleanKind::And(match b {
            BA::True => self,
            BA::False => return BooleanKind::Atom(false.into()).into(),
            BA::And(And { args }) => self.extend(args),
            BA::Dummy | BA::Or(_) | BA::Not(_) | BA::Symbol(_) => self.uneval_insert(b.to_basic()),
        })
        .into()
    }
}

impl CalcursType for And {
    const ID: TypeID = TypeID::And;

    fn to_basic(self) -> Basic {
        BooleanKind::And(self).to_basic()
    }
}

/// Logical Or. Implemented with a list of [Basic]
///
/// The only simplifications performed by the default [Or] are: \
/// (a | b) | (c | d) => a | b | c | d (flatten Or) \
/// a | b | c | a => a | b | c,  \
/// a | b | c | true => false,  \
/// a | b | c | false => a | b | c,
///
#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub struct Or {}

impl Or {
    get_ref_impl!();
}

impl CalcursType for Or {
    const ID: TypeID = TypeID::Or;

    fn to_basic(self) -> Basic {
        BooleanKind::Or(self).to_basic()
    }
}

#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub struct Not {}

impl Not {
    get_ref_impl!();
}

impl CalcursType for Not {
    const ID: TypeID = TypeID::Not;

    fn to_basic(self) -> Basic {
        BooleanKind::Not(self).to_basic()
    }
}

pub const FALSE: Basic = Basic {
    kind: BasicKind::Boolean(Boolean {
        kind: BooleanKind::Atom(BooleanAtom { val: false }),
    }),
};
pub const TRUE: Basic = Basic {
    kind: BasicKind::Boolean(Boolean {
        kind: BooleanKind::Atom(BooleanAtom { val: true }),
    }),
};

#[cfg(test)]
mod boolean_types {

    use crate::*;

    #[test]
    fn ast_getter() {
        const _: bool = TRUE.is::<Basic>();
        assert!(TRUE.is::<Basic>());
        assert!(TRUE.is::<Boolean>());
        assert!(TRUE.is::<BooleanAtom>());

        assert!(!FALSE.is::<Symbol>());
        assert!(!FALSE.is::<And>());
        assert!(!FALSE.is::<Not>());
        assert!(!FALSE.is::<Or>());

        assert!(TRUE.get_ref::<BooleanAtom>().unwrap().val);
        assert!(!FALSE.get_ref::<BooleanAtom>().unwrap().val);
    }

    #[test]
    fn and() {
        let and = And::and(And::and(TRUE, Symbol::new("x")), FALSE).to_basic();
        assert!(!and.get_ref::<BooleanAtom>().unwrap().val);

        let and = And::and(And::and(TRUE, Symbol::new("x")), Symbol::new("x")).to_basic();
        assert!(!and.get_ref::<Symbol>().is_some());
        assert!(and
            .get_ref::<And>()
            .unwrap()
            .args
            .get(&Symbol::new("x").to_basic())
            .is_some());
    }
}
