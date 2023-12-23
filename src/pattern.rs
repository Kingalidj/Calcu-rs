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
        const Atom     = bit!(0);
        const Symbol   = bit!(1)  | bit!(Atom);
        const Numeric  = bit!(2)  | bit!(Atom);

        const Undef    = bit!(3)  | bit!(Numeric);
        const Inf      = bit!(4)  | bit!(Numeric);
        const Rational = bit!(5)  | bit!(Numeric);

        const Int      = bit!(6)  | bit!(Rational);
        const UOne     = bit!(7)  | bit!(Int);

        const Binary   = bit!(8);
        const Add      = bit!(9)  | bit!(Binary);
        const Mul      = bit!(10) | bit!(Binary);
        const Pow      = bit!(11) | bit!(Binary);

        const Zero     = bit!(12);
        const Pos      = bit!(13);
        const Neg      = bit!(14);

        //const AtomicBinary = bit!(Atom) | bit!(Binary);
        //const AtomicAdd    = bit!(Add)  | bit!(AtomicBinary);

        // compositions

        const One          = bit!(Pos)  | bit!(UOne);
        const PosInt       = bit!(Pos)  | bit!(Int);
        const PosRatio     = bit!(Pos)  | bit!(Rational);
        const PosInf       = bit!(Pos)  | bit!(Inf);

        const MinusOne     = bit!(Neg)  | bit!(UOne);
        const NegInt       = bit!(Neg)  | bit!(Int);
        const NegRatio     = bit!(Neg)  | bit!(Rational);
        const NegInf       = bit!(Neg)  | bit!(Inf);
    }
}

impl From<Item> for Pattern {
    fn from(value: Item) -> Self {
        Pattern::Itm(value)
    }
}

impl Pattern {
    pub const fn to_item(&self) -> Item {
        match self {
            Self::Itm(itm) => *itm,
            Self::Binary { op, .. } => *op,
        }
    }

    #[inline]
    pub const fn is(&self, itm: Item) -> bool {
        let b = itm.bits();
        (self.to_item().bits() & b) == b
    }
}

macro_rules! get_itm {
    (Numeric: $e:expr) => {
        if let crate::base::Base::Numeric(n) = $e {
            n
        } else {
            panic!("get_itm for Numeric failed");
        }
    };

    (Rational: $e:expr) => {
        if let crate::base::Base::Numeric(crate::numeric::Numeric::Rational(r)) = $e {
            r
        } else {
            panic!("get_itm for Rational failed");
        }
    };
}
pub(crate) use get_itm;
