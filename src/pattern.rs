use bitflags::bitflags;

macro_rules! bit {
    ($x:literal) => {
        1 << $x
    };

    ($x:ident) => {
        calcu_rs::pattern::Item::$x.bits()
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
        const Constant = bit!(3)  | bit!(Atom);
        const Scalar   = bit!(4) | bit!(Constant);

        const Undef    = bit!(5)  | bit!(Constant);
        const Inf      = bit!(6)  | bit!(Constant);
        const Rational = bit!(7)  | bit!(Scalar);
        const Float    = bit!(8)  | bit!(Scalar);

        const Integer  = bit!(9)  | bit!(Rational);
        const UOne     = bit!(10)  | bit!(Integer);

        const Binary   = bit!(11);
        const Sum      = bit!(12) | bit!(Binary);
        const Prod     = bit!(13) | bit!(Binary);
        const Pow      = bit!(14) | bit!(Binary);

        const Zero     = bit!(15) | bit!(Integer);
        const Pos      = bit!(16);
        const Neg      = bit!(17);

        //const AtomicBinary = bit!(Atom) | bit!(Binary);
        //const AtomicAdd    = bit!(Add)  | bit!(AtomicBinary);

        // compositions

        const One          = bit!(Pos)  | bit!(UOne);
        const PosInt       = bit!(Pos)  | bit!(Integer);
        const PosRatio     = bit!(Pos)  | bit!(Rational);
        const PosFloat     = bit!(Pos)  | bit!(Float);
        const PosInf       = bit!(Pos)  | bit!(Inf);

        const MinusOne     = bit!(Neg)  | bit!(UOne);
        const NegInt       = bit!(Neg)  | bit!(Integer);
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
