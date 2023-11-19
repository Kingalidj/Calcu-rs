use crate::{
    base::{Base, CalcursType, Symbol, PTR},
    numeric::constants::{ONE, ZERO},
    operator::{Add, Mul, Pow, Sub},
    pattern::pat,
};

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Derivative {
    pub deriv: PTR<Base>,
    pub indep: Symbol,
    pub degree: u32,
}

impl CalcursType for Derivative {
    #[inline(always)]
    fn base(self) -> Base {
        Base::Derivative(self)
    }
}

impl Derivative {
    /// d(f) / d(x)
    pub fn apply(f: Base, x: &Symbol) -> Base {
        use Base as B;

        match f {
            // d(n) / d(x) => 0
            B::Numeric(_) => ZERO.base(),

            // d(x) / d(x) => 1
            // d(y) / d(x) => 0
            B::Symbol(ref sym) => if sym == x { ONE } else { ZERO }.base(),

            // sum rule
            B::Add(add) => Self::apply_sum(add, x),

            // chain rule
            B::Mul(mul) => Self::apply_chain(mul, x),

            // power rule
            B::Pow(pow) => Self::apply_pow(*pow, x),

            B::Derivative(mut d) if &d.indep == x => {
                d.degree += 1;
                d.base()
            }

            f => Derivative {
                deriv: f.base().into(),
                indep: x.clone(),
                degree: 1,
            }
            .base(),
        }
    }

    pub fn subs(self, _dict: &crate::base::SubsDict) -> Base {
        //TODO: subs in derivative?
        self.base()
    }

    /// d(f + g) / d(x) => d(f) / d(x) + d(g) / d(x)
    ///
    /// apply summation rule
    fn apply_sum(add: Add, x: &Symbol) -> Base {
        let mut sum = ZERO.base();
        for mul in add.args.into_mul_iter() {
            sum += Derivative::apply_chain(mul, x);
        }
        sum
    }

    /// d(f * g) / d(x) => g * d(f) / d(x) + f * d(g) / d(x)
    ///
    /// apply chain rule
    fn apply_chain(mul: Mul, x: &Symbol) -> Base {
        // d(n * f) / d(x) => n * d(f) / d(x)
        let coeff = mul.coeff;
        let args: Vec<_> = mul.args.into_pow_iter().collect();

        let mut derivs = vec![];

        for pow in args.clone() {
            let deriv = Derivative::apply_pow(pow, x);
            derivs.push(deriv);
        }

        let mut sum = ZERO.base();

        // (f * g * h * ...)' => f' * g * h * ... + f * g' * h * ... + ...
        for (i, deriv) in derivs.into_iter().enumerate() {
            let mut prod = deriv;

            for (j, a) in args.clone().into_iter().enumerate() {
                if i == j {
                    continue;
                }

                prod *= a.base();
            }
            sum += prod;
        }

        Mul::mul(coeff, sum)
    }

    // d(f^g) / d(x)
    //
    // apply power rule
    fn apply_pow(p: Pow, x: &Symbol) -> Base {
        pat!(use);

        match (p.base, p.exp) {
            // n^m => 0
            (pat!(Numeric), pat!(Numeric)) => ZERO.base(),

            // f^n => n * f^(n - 1)
            (f, pat!(Rational: n)) if !n.is_zero() => {
                Mul::mul(n, Pow::pow(f, Sub::sub(n, ONE))).base()
            }

            // TODO: 1 / f
            // TODO: (f / g)' => (g*f' - f*g') / g^2

            //  no further rules...
            (base, exp) => Derivative {
                deriv: Pow { base, exp }.base().into(),
                indep: x.clone(),
                degree: 1,
            }
            .base(),
        }
    }
}
