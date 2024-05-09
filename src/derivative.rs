/*
use crate::numeric::Infinity;
use crate::{
    base::{Expr, CalcursType, Symbol},
    operator::{Sum, Prod, Pow},
    rational::Rational,
};
use calcu_rs::numeric::Float;

pub trait Differentiable: CalcursType {
    type Output;
    fn derive(self, indep: &str) -> Self::Output;
}

impl Differentiable for Sum {
    type Output = Expr;

    // f + g = f' + g'
    fn derive(self, _indep: &str) -> Self::Output {
        todo!()
        //let mut sum = Add::zero();

        //for mul in self.sum.into_mul_iter() {
        //    sum.arg(mul.derive(indep));
        //}

        //sum.reduce()
    }
}
impl Differentiable for Prod {
    type Output = Expr;

    // f * g = f'*g + f*g'
    fn derive(self, _indep: &str) -> Self::Output {
        //let mut sum = Add::zero();

        //let args: Vec<_> = self.product.into_pow_iter().collect();
        //let coeff = self.coeff;

        //for i in 0..args.len() {
        //    let mut prod = Mul::zero();
        //    prod.coeff = coeff;

        //    let deriv = args.get(i).unwrap().clone().derive(indep);
        //    prod.arg(deriv);

        //    for j in 0..args.len() {
        //        if i != j {
        //            let a = args.get(j).unwrap().clone().base();
        //            prod.arg(a);
        //        }
        //    }
        //    sum.arg(prod.reduce());
        //}
        //sum.reduce()
        todo!()
    }
}
impl Differentiable for Pow {
    type Output = Expr;

    // derive f^g
    fn derive(self, _indep: &str) -> Self::Output {
        todo!()
        //let b = self.base.desc();
        //let e = self.exp.desc();

        //identity! { (b, e) {
        //    // x^n -> n * x^(n-1)
        //    (Item::Symbol, Item::Numeric) => {
        //        //let n = get_itm!(Numeric: self.exp);
        //        let n = self.exp;
        //        let x = get_itm!(Symbol: self.base);
        //        if x.name == indep {
        //            n.clone() * x.base().pow(n - Rational::one().base())
        //        } else {
        //            Rational::zero().base()
        //        }
        //    },

        //    // f^n -> n * f^(n-1) * f'
        //    (_, Item::Numeric) => {
        //       let n: Numeric = self.exp.try_into().expect("numeric type");
        //       let f = self.base;
        //       let df = f.clone().derive(indep);
        //       n.base() * f.pow(n + Rational::minus_one().num()) * df
        //    },

        //    // f(x)^g(x)
        //    default => unimplemented!("can't derive this function")
        //}}
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
impl Differentiable for Symbol {
    type Output = Rational;

    fn derive(self, indep: &str) -> Self::Output {
        (&self).derive(indep)
    }
}
impl Differentiable for Expr {
    type Output = Expr;

    fn derive(self, indep: &str) -> Self::Output {
        use Expr as B;
        match self {
            B::Symbol(s) => s.derive(indep).into()
            //B::Numeric(n) => n.derive(indep).base(),
            B::Sum(a) => a.derive(indep),
            B::Prod(m) => m.derive(indep),
            B::Pow(p) => p.derive(indep),
            B::Float(_) | B::Infinity(_) | B::Rational(_) => Rational::zero().base(),
            B::Undefined => B::Undefined,
        }
    }
}
impl Differentiable for Float {
    type Output = Rational;
    fn derive(self, _: &str) -> Self::Output {
        Rational::zero()
    }
}
impl Differentiable for Infinity {
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

 */
