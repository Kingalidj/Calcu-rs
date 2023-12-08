macro_rules! itm {

    // Rational patterns

    (ratio: 1) => {
        itm!(rational: Rational {
            numer: 1,
            denom: itm!(rational: NonZero { non_zero_val: 1 }),
            sign: itm!(numeric: Sign::Positive),
            expon: 0
        })
    };

    (ratio: -1) => {
        itm!(rational: Rational {
            numer: 1,
            denom: itm!(rational: NonZero { non_zero_val: 1 }),
            sign: itm!(numeric: Sign::Negative),
            expon: 0
        })
    };

    (ratio: 0) => {
        itm!(rational: Rational {
            numer: 0,
            denom: itm!(rational: NonZero { non_zero_val: 1 }),
            sign: itm!(numeric: Sign::Positive),
            expon: 0
        })
    };

    (ratio: +) => {
        itm!(rational: Rational {
            sign: itm!(numeric: Sign::Positive),
            ..
        })
    };

    (ratio: -) => {
        itm!(rational: Rational {
            sign: itm!(numeric: Sign::Negative),
            ..
        })
    };

    // Numeric patterns:

    // Numeric::Rational(_)
    (num: Rational: $($n:tt)+) => { itm!(numeric: Numeric::Rational($($n)+))};

   // Numeric::Infinity(_)
   (num: Infinity: $($n:tt)+) => {itm!(numeric: Numeric::Infinity($($n)+))};

    (num: 0) => { itm!(num::ratio: 0) };
    (num: 1) => { itm!(num::ratio: 1) };
    (num: -1) => { itm!(num::ratio: -1) };
    (num: +oo) => { itm!(numeric: Numeric::Infinity(itm!(numeric: Infinity { sign: itm!(numeric: Sign::Positive) }))) };
    (num: -oo) => { itm!(numeric: Numeric::Infinity(itm!(numeric: Infinity { sign: itm!(numeric: Sign::Negative) }))) };

    (num: oo) => { itm!(num: Infinity { .. }) };
    (num: undef) => { itm!(numeric: Numeric::Undefined(_) )};
    (num: +) => { itm!(num::ratio: +) };
    (num: -) => { itm!(num::ratio: -) };

    // Base patterns:

    (Rational: $($n:tt)+) => { itm!(base::num: Rational: $($n)+ )};
    (Numeric: $($n:tt)+) => {crate::base::Base::Numeric($($n)+)};

    (0) => { itm!(base::num: 0) };
    (1) => { itm!(base::num: 1) };
    (-1) => { itm!(base::num: -1) };
    (+) => { itm!(base::num: +) };
    (-) => { itm!(base::num: -) };
    (+oo) => { itm!(base::num: +oo) };
    (-oo) => { itm!(base::num: -oo) };

    (oo) => { itm!(base: num: oo) };
    (undef) => {itm!(base::num: undef) };



    (rational: $($n: tt)+) => {
        crate::rational::$($n)+
    };

    (num::ratio: $($n: tt)+) => {
        crate::numeric::Numeric::Rational(itm!(ratio: $($n)+))
    };

    // numeric -> module crate::numeric
    // num -> enum Numeric
    (numeric: $($n: tt)+) => {
        crate::numeric::$($n)+
    };

    (base::num: $($n: tt)+) => {
        crate::base::Base::Numeric(itm!(num: $($n)+))
    };

    //(base: $($n: tt)+) => {
    //    crate::base::$($n)+
    //};

}

pub(crate) use itm;
