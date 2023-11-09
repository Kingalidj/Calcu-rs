macro_rules! pat {
    (use) => {
        #[allow(unused_imports)]
        use crate::{base::Base, numeric::{Number, Infinity, Undefined, Sign}, rational::{Rational, NonZeroUInt}};
    };

    (Number: 0) => {
        Number::Rational(Rational { numer: 0, .. })
    };

    (Number: 1) => {
        Number::Rational(Rational { numer: 1, denom: NonZeroUInt(1), sign: Sign::Positive })
    };

    (Number: -1) => {
        Number::Rational(Rational { numer: 1, denom: NonZeroUInt(1), sign: Sign::Negative })
    };

    (Number: undef) => {
        Number::Undefined(_)
    };

    (Number: +) => {
        Number::Infinity(Infinity { sign: Sign::Positive })
        | Number::Rational(Rational { sign: Sign::Positive, ..})
    };

    (Number: -) => {
        Number::Infinity(Infinity { sign: Sign::Negative })
        | Number::Rational(Rational { sign: Sign::Negative, ..})
    };

    (0) => {
        Base::Number(pat!(Number: 0))
    };

    (1) => {
        Base::Number(pat!(Number: 1))
    };

    (-1) => {
        Base::Number(pat!(Number: -1))
    };

    (undef) => {
        Base::Number(pat!(Number: undef))
    };

    (Rational: $r: ident) => {
        Base::Number(Number::Rational($r))
    };

    (Number: $n: ident) => {
        Base::Number($n)
    };

    // +Inf, +rational
    (+) => {
        Base::Number(pat!(Number: +))
    };

    // -Inf, -rational
    (-) => {
        Base::Number(pat!(Number: -))
    };
}

pub(crate) use pat;
