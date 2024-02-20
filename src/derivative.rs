use crate::{
    base::{Base, CalcursType, Differentiable, Symbol},
    numeric::Numeric,
    operator::{Add, Mul, Pow},
    pattern::{get_itm, Item},
    rational::Rational,
};

impl Differentiable for Add {
    type Output = Base;

    // f + g = f' + g'
    fn derive(self, indep: &str) -> Self::Output {
        let mut sum = Add::new_raw();

        for mul in self.sum.into_mul_iter() {
            sum.arg(mul.derive(indep));
        }

        sum.reduce()
    }
}

impl Differentiable for Mul {
    type Output = Base;

    // f * g = f'*g + f*g'
    fn derive(self, indep: &str) -> Self::Output {
        let mut sum = Add::new_raw();

        let args: Vec<_> = self.product.into_pow_iter().collect();
        let coeff = self.coeff;

        for i in 0..args.len() {
            let mut prod = Mul::new_raw();
            prod.coeff = coeff;

            let deriv = args.get(i).unwrap().clone().derive(indep);
            prod.arg(deriv);

            for j in 0..args.len() {
                if i != j {
                    let a = args.get(j).unwrap().clone().base();
                    prod.arg(a);
                }
            }
            sum.arg(prod.reduce());
        }
        sum.reduce()
    }
}

impl Differentiable for Pow {
    type Output = Base;

    // derive f^g
    fn derive(self, indep: &str) -> Self::Output {
        let b = self.base.desc();
        let e = self.exp.desc();

        if b.is(Item::Numeric) && e.is(Item::Numeric) {
            Rational::zero().base()
        } else if b.is(Item::Symbol) && e.is(Item::Numeric) {
            // x^n -> n * x^(n-1)
            let n = get_itm!(Numeric: self.exp);
            let x = get_itm!(Symbol: self.base);
            if x.name == indep {
                n.base() * x.base().pow(n - Rational::one().num())
            } else {
                Rational::zero().base()
            }
        } else if e.is(Item::Numeric) {
            // f^n -> n * f^(n-1) * f'
            let n = get_itm!(Numeric: self.exp);
            let f = self.base;
            let df = f.clone().derive(indep);
            n.base() * f.pow(n + Rational::minus_one().num()) * df
        } else {
            unimplemented!("can't derive this function")
        }
    }
}

impl Differentiable for &Symbol {
    type Output = Rational;

    fn derive(self, indep: &str) -> Self::Output {
        if self.name == indep {
            Rational::one()
        } else {
            Rational::zero()
        }
    }
}

impl Differentiable for Base {
    type Output = Base;

    fn derive(self, indep: &str) -> Self::Output {
        use Base as B;
        match self {
            B::Symbol(s) => s.derive(indep).base(),
            B::Numeric(n) => n.derive(indep).base(),
            B::Add(a) => a.derive(indep),
            B::Mul(m) => m.derive(indep),
            B::Pow(p) => p.derive(indep),
        }
    }
}

impl Differentiable for Numeric {
    type Output = Rational;
    fn derive(self, _: &str) -> Self::Output {
        Rational::zero()
    }
}

impl Differentiable for Rational {
    type Output = Rational;
    fn derive(self, _: &str) -> Self::Output {
        Rational::zero()
    }
}
