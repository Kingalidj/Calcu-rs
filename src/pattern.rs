use bitflags::bitflags;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd)]
pub enum Pattern {
    Itm(Item),
    Binary { lhs: Item, op: Item, rhs: Item },
}

macro_rules! bit {
    ($x:literal) => {
        1 << $x
    };

    ($x:ident) => {
        crate::pattern::Item::$x.bits()
    };
}

bitflags! {
    /// basic description of an item for pattern matching
    #[rustfmt::skip]
    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
    pub struct Item: u32 {
        const All      = bit!(0);
        const Atom     = bit!(1);
        const Symbol   = bit!(2)  | bit!(Atom);
        const Constant  = bit!(3)  | bit!(Atom);
        const Finite    = bit!(4) | bit!(Constant);

        const Undef    = bit!(5)  | bit!(Constant);
        const Inf      = bit!(6)  | bit!(Constant);
        const Rational = bit!(7)  | bit!(Finite);
        const Float    = bit!(8)  | bit!(Finite);

        const Int      = bit!(9)  | bit!(Rational);
        const UOne     = bit!(10)  | bit!(Int);

        const Binary   = bit!(11);
        const Sum      = bit!(12) | bit!(Binary);
        const Prod      = bit!(13) | bit!(Binary);
        const Pow      = bit!(14) | bit!(Binary);

        const Zero     = bit!(15) | bit!(Int);
        const Pos      = bit!(16);
        const Neg      = bit!(17);

        //const AtomicBinary = bit!(Atom) | bit!(Binary);
        //const AtomicAdd    = bit!(Add)  | bit!(AtomicBinary);

        // compositions

        const One          = bit!(Pos)  | bit!(UOne);
        const PosInt       = bit!(Pos)  | bit!(Int);
        const PosRatio     = bit!(Pos)  | bit!(Rational);
        const PosFloat     = bit!(Pos)  | bit!(Float);
        const PosInf       = bit!(Pos)  | bit!(Inf);

        const MinusOne     = bit!(Neg)  | bit!(UOne);
        const NegInt       = bit!(Neg)  | bit!(Int);
        const NegRatio     = bit!(Neg)  | bit!(Rational);
        const NegFloat     = bit!(Neg)  | bit!(Float);
        const NegInf       = bit!(Neg)  | bit!(Inf);
    }
}

impl Item {
    pub const fn is(&self, itm: Item) -> bool {
        let b = itm.bits();
        (self.bits() & b) == b
    }

    pub const fn is_not(&self, itm: Item) -> bool {
        !self.is(itm)
    }
}

impl From<Item> for Pattern {
    fn from(value: Item) -> Self {
        Pattern::Itm(value)
    }
}

impl Pattern {
    pub const fn get_item(&self) -> Item {
        match self {
            Self::Itm(itm) => *itm,
            Self::Binary { op, .. } => *op,
        }
    }

    #[inline(always)]
    pub const fn is(&self, itm: Item) -> bool {
        let b = itm.bits();
        (self.get_item().bits() & b) == b
    }

    #[inline(always)]
    pub const fn is_not(&self, itm: Item) -> bool {
        !self.is(itm)
    }
}
