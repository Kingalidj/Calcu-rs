pub(crate) mod __ {
    pub(crate) mod p {
        #[allow(unused_imports)]
        pub(crate) use crate::{
            base::Base,
            numeric::{Infinity, Numeric, Sign, Undefined},
            pattern::num_pat,
            rational::{NonZeroUInt, Rational},
        };
    }
}

macro_rules! num_pat {
    (base: $($n: tt)*) => {
        p::Base::Numeric(p::num_pat!($($n)*))
    };

    (0) => {
        p::Numeric::Rational(p::Rational { numer: 0, .. })
    };

    (1) => {
        p::Numeric::Rational(p::Rational {
            numer: 1,
            denom: p::NonZeroUInt { non_zero_val: 1 },
            sign: p::Sign::Positive,
        })
    };

    (-1) => {
        p::Numeric::Rational(p::Rational {
            numer: 1,
            denom: p::NonZeroUInt { non_zero_val: 1 },
            sign: p::Sign::Negative,
        })
    };

    (undef) => {
        p::Numeric::Undefined(_)
    };

    (+) => {
        p::Numeric::Rational(p::Rational {
            sign: p::Sign::Positive,
            ..
        })
    };

    (-) => {
        p::Numeric::Rational(p::Rational {
            sign: p::Sign::Negative,
            ..
        })
    };

    (Infinity: $i: ident) => {
        p::Numeric::Infinity($i)
    };


    (+oo) => {
        p::Numeric::Infinity(p::Infinity{ sign: p::Sign::Positive })
    };

    (-oo) => {
        p::Numeric::Infinity(p::Infinity{ sign: p::Sign::Negative })
    };

    ($n: ident) => {
        p::Base::Numeric($n)
    };

    () => {
        p::Base::Numeric(_)
    };

    (Rational: $r: tt) => {
        p::Numeric::Rational($r)
    }
}

macro_rules! pat {
    (use) => {
        use crate::pattern::__::p;
        #[allow(unused_imports)]
        use crate::pattern::num_pat;
    };

    // --- types ---

    (Symbol: $s: ident) => {
        p::Base::Symbol($s)
    };

    (Numeric) => {
        p::num_pat!()
    };

    (Numeric: $($n: tt)+) => {
        p::num_pat!($($n)+)
    };

    (Rational: $r: ident) => {
        p::num_pat!(base: Rational: $r)
    };

    // --- values ---

    (0) => {
        p::num_pat!(base: 0)
    };

    (1) => {
        p::num_pat!(base: 1)
    };

    (-1) => {
        p::num_pat!(base: -1)
    };

    (undef) => {
        p::num_pat!(base: undef)
    };

    (+) => {
        p::num_pat!(base: +)
    };

    (-) => {
        p::num_pat!(base: -)
    };
}

pub(crate) use num_pat;
pub(crate) use pat;
