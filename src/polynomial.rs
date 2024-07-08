use std::{
    collections::BTreeMap,
    ops
};
use std::fmt::{Debug};
use crate::*;

type Exponent = Rational;
type Coefficient = Rational;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum GME {
    Var(Symbol),
}

/*
#[derive(Clone, PartialOrd, Ord)]
pub struct PowerProd {
    /// [Symbol]^[Rational] (assumption: symbol is non zero)
    pub(crate) prod: BTreeMap<Symbol, Rational>,
}
impl Debug for PowerProd {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        if self.prod.is_empty() {
            return write!(f, "1");
        }
        self.prod.iter().try_for_each(|(s, r)| {
            write!(f, "{s}^{r}")
        })
    }
}
impl<const N: usize> From<[(Symbol, Rational); N]> for PowerProd {
    fn from(arr: [(Symbol, Rational); N]) -> Self {
        Self { prod: arr.into() }
    }
}
impl From<Symbol> for PowerProd {
    fn from(sym: Symbol) -> Self {
        PowerProd::from([(sym, Rational::ONE)])
    }
}
impl Eq for PowerProd {}
impl PartialEq for PowerProd {
    fn eq(&self, other: &Self) -> bool {
        self.vars().eq(other.vars())
    }
}
impl PowerProd {
    fn mul_var(&mut self, var: Symbol, exp: Rational) {
        if exp.is_zero() {
            return;
        }
        self.prod
            .entry(var)
            .and_modify(|e| *e += exp.clone())
            .or_insert(exp);
    }

    fn vars(&self) -> impl Iterator<Item = (&Symbol, &Rational)> {
        self.prod.iter().filter(|(_, r)| !r.is_zero())
    }
}
impl ops::Mul for PowerProd {
    type Output = PowerProd;

    fn mul(mut self, mut rhs: Self) -> Self::Output {
        if self.prod.len() < rhs.prod.len() {
            std::mem::swap(&mut self, &mut rhs);
        }
        rhs.prod.into_iter().for_each(|(v, e)| {
            self.mul_var(v, e);
        });
        self
    }
}
impl ops::Add for PowerProd {
    type Output = Polynomial;

    fn add(self, rhs: Self) -> Self::Output {
        Polynomial::from(self) + Polynomial::from(rhs)
    }
}
impl ops::Add<Rational> for PowerProd {
    type Output = Polynomial;

    fn add(self, rhs: Rational) -> Self::Output {
        let mut p = Polynomial::from(self);
        p.coeff = rhs;
        p
    }
}
impl ops::Sub for PowerProd {
    type Output = Polynomial;

    fn sub(self, rhs: Self) -> Self::Output {
        Polynomial::from(self) - Polynomial::from(rhs)
    }
}
impl ops::Mul<Rational> for PowerProd {
    type Output = Polynomial;

    fn mul(self, rhs: Rational) -> Self::Output {
        Polynomial::new(Rational::ONE, [(self, rhs)])
    }
}
impl Pow<Rational> for PowerProd {
    type Output = Self;

    fn pow(mut self, rhs: Rational) -> Self::Output {
        self.prod.iter_mut().for_each(|(v, r)| {
            *r *= rhs.clone();
        });
        self
    }
}
#[derive(Default, Clone, PartialOrd, Ord)]
pub struct Polynomial {
    coeff: Rational,
    /// [Rational] * [PowerProd]
    sum: BTreeMap<PowerProd, Rational>,
}
impl Debug for Polynomial {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.coeff)?;
        if self.sum.is_empty() {
            return Ok(());
        }
        self.sum.iter().try_for_each(|(m, c)| {
            write!(f, " + {c}{:?}", m)
        })
    }
}
impl Eq for Polynomial {}
impl PartialEq for Polynomial {
    fn eq(&self, other: &Self) -> bool {
        self.coeff == other.coeff && self.monoms().eq(other.monoms())
    }
}
impl From<Rational> for Polynomial {
    fn from(coeff: Rational) -> Self {
        Self { coeff, sum: Default::default() }
    }
}
impl From<PowerProd> for Polynomial {
    fn from(m: PowerProd) -> Self {
        Self { coeff: Rational::ZERO, sum: [(m, Rational::ONE)].into() }
    }
}
impl Polynomial {
    pub fn new<const N: usize>(coeff: Rational, sum: [(PowerProd, Rational); N]) -> Self {
        Self { coeff, sum: sum.into() }
    }

    pub fn monoms(&self) -> impl Iterator<Item = (&PowerProd, &Rational)> {
        self.sum.iter().filter(|(_, r)| !r.is_zero())
    }
    pub fn into_monoms(self) -> impl Iterator<Item = (PowerProd, Rational)> {
        self.sum.into_iter().filter(|(_, r)| !r.is_zero())
    }

    pub fn add_monom(&mut self, m: PowerProd, coeff: Rational) {
        if coeff.is_zero() {
            return
        }
        self.sum.entry(m)
            .and_modify(|c| *c += coeff.clone())
            .or_insert(coeff);
    }
    pub fn sub_monom(&mut self, m: PowerProd, coeff: Rational) {
        self.add_monom(m, coeff * Rational::MINUS_ONE)
    }
}
impl ops::Add for Polynomial {
    type Output = Polynomial;

    fn add(mut self, rhs: Self) -> Self::Output {
        self += rhs;
        self
    }
}
impl ops::AddAssign for Polynomial {
    fn add_assign(&mut self, rhs: Self) {
        self.coeff += rhs.coeff;
        rhs.sum.into_iter().for_each(|(m, c)| {
            self.add_monom(m, c)
        });
    }
}
impl ops::Add<Rational> for Polynomial {
    type Output = Polynomial;

    fn add(mut self, rhs: Rational) -> Self::Output {
        self.coeff += rhs;
        self
    }
}
impl ops::Add<PowerProd> for Polynomial {
    type Output = Polynomial;

    fn add(mut self, rhs: PowerProd) -> Self::Output {
        self.sum.entry(rhs).and_modify(|e| *e += Rational::ONE)
            .or_insert(Rational::ONE);
        self
    }
}
impl ops::Sub for Polynomial {
    type Output = Polynomial;

    fn sub(mut self, rhs: Self) -> Self::Output {
        self.coeff -= rhs.coeff;
        rhs.sum.into_iter().for_each(|(m, c)| {
            self.sub_monom(m, c)
        });
        self
    }
}
impl ops::Mul<Rational> for Polynomial {
    type Output = Polynomial;

    fn mul(mut self, rhs: Rational) -> Self::Output {
        self *= rhs;
        self
    }
}
impl ops::MulAssign<Rational> for Polynomial {
    fn mul_assign(&mut self, rhs: Rational) {
        self.coeff *= rhs.clone();
        self.sum.iter_mut().for_each(|(_, r)| {
            *r *= rhs.clone()
        });
    }
}
impl ops::Mul<PowerProd> for Polynomial {
    type Output = Polynomial;

    fn mul(mut self, rhs: PowerProd) -> Self::Output {
        self *= rhs;
        self
    }
}
impl ops::MulAssign<PowerProd> for Polynomial {
    fn mul_assign(&mut self, rhs: PowerProd) {
        let mut c = Rational::ZERO;
        std::mem::swap(&mut c, &mut self.coeff);
        self.sum = self.sum.clone().into_iter().filter(|(_, r)| !r.is_zero()).map(|(v, r)| {
            (v * rhs.clone(), r)
        }).collect();
        self.add_monom(rhs, c);
    }
}

impl ops::Mul for Polynomial {
    type Output = Polynomial;

    fn mul(self, rhs: Self) -> Self::Output {
        let mut result = Polynomial::from(self.coeff.clone() * rhs.coeff.clone());

        for (m1, c1) in self.sum.into_iter() {
            for (m2, c2) in rhs.sum.iter() {
                result.add_monom(m1.clone() * m2.clone(), c1.clone() * c2.clone());
            }
            result.add_monom(m1, c1 * rhs.coeff.clone());
        }

        for (m2, c2) in rhs.sum.into_iter() {
            result.add_monom(m2, c2 * self.coeff.clone());
        }

        result
    }
}

#[cfg(test)]
mod polynomial_test {
    use super::*;

    macro_rules! p {
        ($n:literal) => {
            Rational::from($n)
        };
        ($($x:ident^$n:literal)*) => {
            PowerProd::from([$((symbol(stringify!($x)), Rational::from($n)), )*])
        }
    }

    fn symbol(s: &'static str) -> Symbol {
        assert!(SymbolTable::is_global());
        SymbolTable::new().insert(s)
    }

    use Polynomial as P;
    use Rational as R;

    #[test]
    fn polynomial() {
        assert_eq!(p!(x^2) + p!(x^1), P::new(R::ZERO, [(p!(x^2), R::ONE), (p!(x^1), R::ONE)]));
        assert_eq!(p!(x^2) + p!(x^2), P::new(R::ZERO, [(p!(x^2), R::TWO)]));
        assert_eq!(p!(x^2) - p!(x^2), P::from(R::ZERO));
        assert_eq!((p!(x^2) + p!(4) + p!(y^2)) * p!(2), P::new(R::from(8), [(p!(x^2), R::TWO), (p!(y^2), Rational::TWO)]));
        // (x^2 + 2) * (y^2 + 1) -> x^2y^2 + x^2 + 2y^2 + 2
        assert_eq!((p!(x^2) + R::TWO) * p!(y^2), Polynomial::new(R::ZERO, [(p!(x^2 y^2), R::ONE), (p!(y^2), R::TWO)]));
        assert_eq!((p!(x^2) + R::TWO) * (p!(y^2) + R::ONE), Polynomial::new(R::TWO, [(p!(x^2 y^2), R::ONE), (p!(x^2), R::ONE), (p!(y^2), R::TWO)]))
    }

    #[test]
    fn power_prod() {
        assert_eq!(p!(x^2) * p!(x^3), p!(x^5));
        assert_eq!(p!(x^2) * p!(x^-2), p!(x^0));
        assert_eq!(p!(x^2) * p!(x^-2), p!(y^0));
        assert_eq!(p!(y^2) * p!(z^-3), p!(y^2 z^-3));
        assert_eq!(p!(x^2).pow(p!(3)), p!(x^6));
    }
}
 */