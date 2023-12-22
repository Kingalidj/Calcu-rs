macro_rules! itm {

    // Rational patterns

    (ratio: 1) => {
        crate::rational::Rational {
            numer: 1,
            denom: crate::rational::NonZero { non_zero_val: 1 },
            sign: crate::numeric::Sign::Positive,
            expon: 0
        }
    };

    (ratio: -1) => {
        crate::rational::Rational {
            numer: 1,
            denom: crate::rational::NonZero { non_zero_val: 1 },
            sign: crate::numeric::Sign::Negative,
            expon: 0
        }
    };

    (ratio: 0) => {
        crate::rational::Rational {
            numer: 0,
            denom: crate::rational::NonZero { non_zero_val: 1 },
            sign: crate::numeric::Sign::Positive,
            expon: 0
        }
    };

    (ratio: +) => {
        crate::rational::Rational {
            sign: crate::numeric::Sign::Positive,
            ..
        }
    };

    (ratio: -) => {
        crate::rational::Rational {
            sign: crate::numeric::Sign::Negative,
            ..
        }
    };

    // Numeric patterns:

    // Numeric::Rational(_)
    (num: Rational: $($n:tt)+) => { crate::numeric::Numeric::Rational($($n)+)};

    // Numeric::Infinity(_)
    (num: Infinity: $($n:tt)+) => {crate::numeric::Numeric::Infinity($($n)+)};

    (num: 0) => { itm!(num_ratio: 0) };
    (num: 1) => { itm!(num_ratio: 1) };
    (num: -1) => { itm!(num_ratio: -1) };
    (num: +) => { itm!(num_ratio: +) };
    (num: -) => { itm!(num_ratio: -) };

    (num: +oo) => { crate::numeric::Numeric::Infinity(crate::numeric::Infinity { sign: crate::numeric::Sign::Positive }) };
    (num: -oo) => { crate::numeric::Numeric::Infinity(crate::numeric::Infinity { sign: crate::numeric::Sign::Negative }) };
    (num: oo) => { crate::numeric::Infinity { .. } };
    (num: undef) => { crate::numeric::Numeric::Undefined(_) };

    // Base patterns:

    (Rational: $($n:tt)+) => { itm!(base_num: Rational: $($n)+ )};
    (Numeric: $($n:tt)+) => {crate::base::Base::Numeric($($n)+)};

    (0) => { itm!(base_num: 0) };
    (1) => { itm!(base_num: 1) };
    (-1) => { itm!(base_num: -1) };
    (+) => { itm!(base_num: +) };
    (-) => { itm!(base_num: -) };
    (+oo) => { itm!(base_num: +oo) };
    (-oo) => { itm!(base_num: -oo) };
    (oo) => { itm!(base: num: oo) };
    (undef) => {itm!(base_num: undef) };


    // switch namespace
    (num_ratio: $($n: tt)+) => {
        crate::numeric::Numeric::Rational(itm!(ratio: $($n)+))
    };


    (base_num: $($n: tt)+) => {
        crate::base::Base::Numeric(itm!(num: $($n)+))
    };

}

pub(crate) use itm;
