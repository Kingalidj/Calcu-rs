macro_rules! itm {

    // Rational patterns

    (ratio: 1) => {
        itm!(ratio: Rational {
            numer: 1,
            denom: itm!(ratio: NonZero { non_zero_val: 1 }),
            sign: itm!(num: Sign::Positive),
            expon: 0
        })
    };

    (ratio: -1) => {
        itm!(ratio: Rational {
            numer: 1,
            denom: itm!(ratio: NonZero { non_zero_val: 1 }),
            sign: itm!(num: Sign::Negative),
            expon: 0
        })
    };

    (ratio: 0) => {
        itm!(ratio: Rational {
            numer: 0,
            denom: itm!(ratio: NonZero { non_zero_val: 1 }),
            sign: itm!(num: Sign::Positive),
            expon: 0
        })
    };

    (ratio: +) => {
        itm!(ratio: Rational {
            sign: itm!(num: Sign::Positive),
            ..
        })
    };

    (ratio: -) => {
        itm!(ratio: Rational {
            sign: itm!(num: Sign::Negative),
            ..
        })
    };

    // Numeric patterns:

    // Numeric::Rational(_)
    (num: Rational: $($n:tt)+) => { itm!(num: Numeric::Rational($($n)+))};

    // Numeric::Infinity(_)
    (num: Infinity: $($n:tt)+) => {itm!(num: Numeric::Infinity($($n)+))};

    (num: 0) => { itm!(num: ratio: 0) };
    (num: 1) => { itm!(num: ratio: 1) };
    (num: -1) => { itm!(num: ratio: -1) };
    (num: +oo) => { itm!(num: Numeric::Infinity(itm!(num: Infinity { sign: itm!(num: Sign::Positive) }))) };
    (num: -oo) => { itm!(num: Numeric::Infinity(itm!(num: Infinity { sign: itm!(num: Sign::Negative) }))) };

    (num: oo) => { itm!(num: Infinity { .. }) };
    (num: undef) => { itm!(num: Numeric::Undefined(_) )};
    (num: +) => { itm!(num: ratio: +) };
    (num: -) => { itm!(num: ratio: -) };

    // Base patterns:

    (Rational: $($n:tt)+) => { itm!(base: num: Rational: $($n)+ )};
    (Numeric: $($n:tt)+) => {itm!(base: Base::Numeric($($n)+))};

    (0) => { itm!(base: num: 0) };
    (1) => { itm!(base: num: 1) };
    (-1) => { itm!(base: num: -1) };
    (+) => { itm!(base: num: +) };
    (-) => { itm!(base: num: -) };
    (+oo) => { itm!(base: num: +oo) };
    (-oo) => { itm!(base: num: -oo) };

    (oo) => { itm!(base: num: oo) };
    (undef) => {itm!(base: num: undef) };

    (ratio: $($n: tt)+) => {
        crate::rational::$($n)+
    };

    (num: ratio: $($n: tt)+) => {
        crate::numeric::Numeric::Rational(itm!(ratio: $($n)+))
    };

    (num: $($n: tt)+) => {
        crate::numeric::$($n)+
    };

    (base: num: $($n: tt)+) => {
        crate::base::Base::Numeric(itm!(num: $($n)+))
    };

    (base: $($n: tt)+) => {
        crate::base::$($n)+
    };

}

pub(crate) use itm;
