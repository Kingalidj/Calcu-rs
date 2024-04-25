use bitflags::bitflags;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd)]
pub enum Pattern {
    Itm(Item),
    Binary { lhs: Item, op: Item, rhs: Item },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd)]
pub enum Pattern2<'a> {
    Item(Item),
    Operation { op: Item, elems: &'a [Item] },
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
        const Numeric  = bit!(3)  | bit!(Atom);

        const Undef    = bit!(4)  | bit!(Numeric);
        const Inf      = bit!(5)  | bit!(Numeric);
        const Rational = bit!(6)  | bit!(Numeric);
        const Float    = bit!(7)  | bit!(Numeric);

        const Int      = bit!(8)  | bit!(Rational);
        const UOne     = bit!(9)  | bit!(Int);

        const Binary   = bit!(10);
        const Add      = bit!(11) | bit!(Binary);
        const Mul      = bit!(12) | bit!(Binary);
        const Pow      = bit!(13) | bit!(Binary);

        const Zero     = bit!(14) | bit!(Int);
        const Pos      = bit!(15);
        const Neg      = bit!(16);

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
    pub const fn to_item(&self) -> Item {
        match self {
            Self::Itm(itm) => *itm,
            Self::Binary { op, .. } => *op,
        }
    }

    #[inline(always)]
    pub const fn is(&self, itm: Item) -> bool {
        let b = itm.bits();
        (self.to_item().bits() & b) == b
    }

    #[inline(always)]
    pub const fn is_not(&self, itm: Item) -> bool {
        !self.is(itm)
    }
}

impl<'a> Pattern2<'a> {
    pub const fn to_item(&self) -> Item {
        match self {
            Pattern2::Item(itm) => *itm,
            Pattern2::Operation { op, .. } => *op,
        }
    }

    pub const fn contains(&self, item: Item) -> bool {
        let b = item.bits();
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

    (Symbol: $e:expr) => {
        if let crate::base::Base::Symbol(e) = $e {
            e
        } else {
            panic!("get_itm for Symbol failed");
        }
    };

    (Add: $e:expr) => {
        if let crate::base::Base::Add(e) = $e {
            e
        } else {
            panic!("get_itm for Add failed");
        }
    };

    (Mul: $e:expr) => {
        if let crate::base::Base::Mul(e) = $e {
            e
        } else {
            panic!("get_itm for Mul failed");
        }
    };

    (Pow: $e:expr) => {
        if let crate::base::Base::Pow(e) = $e {
            e
        } else {
            panic!("get_itm for Pow failed");
        }
    };
}
pub(crate) use get_itm;
