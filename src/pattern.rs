pub(crate) mod __ {
    pub(crate) mod p {
        #[allow(unused_imports)]
        pub(crate) use crate::{
            base::Base,
            numeric::{Infinity, Number, Sign, Undefined},
            pattern::num_pat,
            rational::{NonZeroUInt, Rational},
        };
    }
}

macro_rules! num_pat {
    (base: $($n: tt)*) => {
        p::Base::Number(p::num_pat!($($n)*))
    };

    (0) => {
        p::Number::Rational(p::Rational { numer: 0, .. })
    };

    (1) => {
        p::Number::Rational(p::Rational {
            numer: 1,
            denom: p::NonZeroUInt { non_zero_val: 1 },
            sign: p::Sign::Positive,
        })
    };

    (-1) => {
        p::Number::Rational(p::Rational {
            numer: 1,
            denom: p::NonZeroUInt { non_zero_val: 1 },
            sign: p::Sign::Negative,
        })
    };

    (undef) => {
        p::Number::Undefined(_)
    };

    //TODO: +oo?
    (+) => {
        Number::Rational(p::Rational {
            sign: p::Sign::Positive,
            ..
        })
    };

    (-) => {
        Number::Rational(p::Rational {
            sign: p::Sign::Negative,
            ..
        })
    };

    ($n: ident) => {
        p::Base::Number($n)
    };

    () => {
        p::Base::Number(_)
    };

    (Rational: $r: ident) => {
        p::Number::Rational($r)
    }
}

macro_rules! pat {
    (use) => {
        use crate::pattern::__::p;
    };

    // --- types ---

    (Symbol: $s: ident) => {
        p::Base::Symbol($s)
    };

    (Number) => {
        p::num_pat!()
    };

    (Number: $($n: tt)+) => {
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
