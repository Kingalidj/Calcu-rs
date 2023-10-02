
pub const fn is_same<T: CalcursType, U: CalcursType>() -> bool {
    T::ID as u32 == U::ID as u32
}

pub const fn cast_ref<T: CalcursType, U: CalcursType>(r#ref: &T) -> Option<&U> {
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
}

macro_rules! early_ret {
    ($e: expr) => {
        if let Some(ret) = $e {
            return Some(ret)
        }
    }
}

macro_rules! get_ref_impl {
    () => {
        pub const fn get_ref<T: CalcursType>(&self) -> Option<&T> {
            cast_ref::<Self, T>(self)
        }
    };

    ($($x: ident)+) => {
        pub const fn get_ref<T: CalcursType>(&self) -> Option<&T> {
            early_ret!(cast_ref::<Self, T>(self));
            $( early_ret!(self.$x.get_ref::<T>()); )+
            None
        }
    }
}

#[derive(Debug, Clone)]
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
}

#[derive(Debug, Clone)]
pub enum BasicKind {
    Symbol(Symbol),
    Boolean(Boolean),

    Dummy,
}

impl BasicKind {
    pub const fn get_ref<T: CalcursType>(&self) -> Option<&T> {
        use BasicKind as BK;
        match self {
            BK::Symbol(b) => b.get_ref::<T>(),
            BK::Boolean(b) => b.get_ref::<T>(),
            BK::Dummy => None
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Symbol {
    pub name: String,
}

impl CalcursType for Symbol {
    const ID: TypeID = TypeID::Symbol;
}

impl Symbol {
    get_ref_impl!();
}

#[derive(Debug, Clone)]
pub struct Boolean {
    pub kind: BooleanKind,
}

impl CalcursType for Boolean {
    const ID: TypeID = TypeID::Boolean;
}

impl Boolean {
    get_ref_impl!(kind);
}

#[derive(Debug, Clone)]
pub enum BooleanKind {
    Atom(BooleanAtom),

    And(And),
    Or(Or),
    Not(Not),
}

impl BooleanKind {
    pub const fn get_ref<T: CalcursType>(&self) -> Option<&T> {
        use BooleanKind as BK;
        match self {
            BK::Atom(b) => b.get_ref::<T>(),
            BK::And(b) => b.get_ref::<T>(),
            BK::Or(b) => b.get_ref::<T>(),
            BK::Not(b) => b.get_ref::<T>(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BooleanAtom {
    pub val: bool,
}

impl BooleanAtom {
    get_ref_impl!();
}

impl CalcursType for BooleanAtom {
    const ID: TypeID = TypeID::BooleanAtom;
}

#[derive(Debug, Clone)]
pub struct And {}

impl And {
    get_ref_impl!();
}

impl CalcursType for And {
    const ID: TypeID = TypeID::And;
}

#[derive(Debug, Clone)]
pub struct Or {}

impl Or {
    get_ref_impl!();
}

impl CalcursType for Or {
    const ID: TypeID = TypeID::Or;
}

#[derive(Debug, Clone)]
pub struct Not {}

impl Not {
    get_ref_impl!();
}

impl CalcursType for Not {
    const ID: TypeID = TypeID::Not;
}

pub const FALSE: Basic = Basic { kind: BasicKind::Boolean( Boolean { kind: BooleanKind::Atom(BooleanAtom { val: false }) }) };
pub const TRUE: Basic = Basic { kind: BasicKind::Boolean( Boolean { kind: BooleanKind::Atom(BooleanAtom { val: true }) }) };

#[cfg(test)]
mod test {

    use crate::*;

    #[test]
    fn ast_getter() {
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

}
