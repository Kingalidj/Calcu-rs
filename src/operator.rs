use std::fmt;

use crate::{
    base::{Base, CalcursType},
    numeric::{Infinity, Numeric, Undefined},
    pattern::{get_itm, Item, Pattern},
    rational::Rational,
};

/// Represents addition in symbolic expressions
///
/// Implemented with a coefficient and an [Sum]: \
/// coeff + mul1 + mul2 + mul3...
#[derive(Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Add {
    pub(crate) coeff: Numeric,
    pub(crate) sum: Sum,
}

pub type Sub = Add;

impl Add {
    /// n1 + n2
    pub fn add(n1: impl CalcursType, n2: impl CalcursType) -> Base {
        let mut sum = Self::new_raw();
        sum.arg(n1.base());
        sum.arg(n2.base());
        sum.reduce()
    }

    pub fn desc(&self) -> Pattern {
        let op = Item::Add;
        let lhs = self.coeff.desc().to_item();
        let rhs = self.sum.desc().to_item();
        Pattern::Binary { lhs, op, rhs }
    }

    /// n1 + (-1 * n2)
    #[inline]
    pub fn sub(n1: impl CalcursType, n2: impl CalcursType) -> Base {
        Add::add(n1, Mul::mul(Rational::minus_one(), n2))
    }

    pub fn arg(&mut self, b: Base) {
        use Base as B;
        match b {
            B::Numeric(num) => self.coeff += num,
            B::Mul(mul) => self.sum.add(mul.base()),
            B::Add(add) => {
                self.coeff += add.coeff;
                add.sum
                    .into_mul_iter()
                    .for_each(|mul| self.sum.add(mul.base()));
            }

            base @ (B::Symbol(_) | B::Pow(_)) => {
                self.sum.add(base)
            },
        };
    }

    pub fn reduce(self) -> Base {
        let coeff = self.coeff.desc();

        if self.sum.is_empty() {
            self.coeff.base()
        } else if coeff.is(Item::Zero) && self.sum.is_product() {
            let mul: Mul = self.sum.try_into().unwrap();
            mul.base()
        } else {
            self.base()
        }
    }

    pub fn new_raw() -> Add {
        Self {
            coeff: Rational::zero().num(),
            sum: Default::default(),
        }
    }
}

/// Represents multiplication in symbolic expressions
///
/// Implemented with a coefficient and a hashmap: \
/// coeff * pow1 * pow2 * pow3...
#[derive(Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Mul {
    pub(crate) coeff: Numeric,
    pub(crate) product: Product,
}

pub type Div = Mul;

impl Mul {

    fn zero() -> Self {
        Mul {
            coeff: Rational::one().num(),
            product: Product::zero(),
        }
    }

    /// n1 * n2
    pub fn mul(n1: impl CalcursType, n2: impl CalcursType) -> Base {
        let mut prod = Self::new_raw();
        prod.arg(n1.base());
        prod.arg(n2.base());
        prod.reduce()
    }

    /// n1 * (1 / n2)
    #[inline]
    pub fn div(n1: impl CalcursType, n2: impl CalcursType) -> Base {
        Mul::mul(n1, Pow::pow(n2, Rational::minus_one()))
    }

    pub fn arg(&mut self, b: Base) {
        use Base as B;

        if Undefined.base() == b {
            self.coeff = Undefined.into();
            return;
        } else if self.coeff == Rational::zero().num() || Rational::one().base() == b {
            return;
        }

        match b {
            B::Numeric(num) => self.coeff *= num,
            B::Pow(pow) => self.product.mul(pow.base()),
            B::Mul(mul) => {
                self.coeff *= mul.coeff;
                mul.product
                    .into_pow_iter()
                    .for_each(|pow| self.product.mul(pow.base()));
            }
            B::Symbol(_) | B::Add(_) => self.product.mul(b),
        }
    }

    pub fn reduce(self) -> Base {
        let coeff = self.coeff.desc();

        if self.product.is_empty() {
            self.coeff.base()
        } else if coeff.is(Item::Undef) {
            Undefined.base()
        } else if coeff.is(Item::Zero) {
            Rational::zero().base()
        } else if coeff.is(Item::One) && self.product.is_pow() {
            let p: Pow = self.product.try_into().unwrap();
            p.base()
        } else {
            self.base()
        }
    }

    pub fn new_raw() -> Self {
        Self {
            coeff: Rational::one().num(),
            product: Product::default(),
        }
    }

    fn fmt_coeff(coeff: &Numeric, prod_desc: Pattern, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let show_sign = coeff.desc().is(Item::Neg);
        let show_coeff = !coeff.desc().is(Item::UOne);
        let show_op = show_coeff && !prod_desc.is(Item::Pow);

        if show_sign {
            write!(f, "-")?;
        }
        if show_coeff {
            write!(f, "{coeff}")?;
        }

        if show_op {
            write!(f, " * ")?;
        }

        Ok(())
    }

    fn fmt_parts(coeff: &Numeric, prod: &SumElem, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        Self::fmt_coeff(coeff, prod.desc(), f)?;
        write!(f, "{prod}")
    }

    #[inline]
    pub fn desc(&self) -> Pattern {
        let op = Item::Mul;
        let lhs = self.coeff.desc().to_item();
        let rhs = self.product.desc().to_item();
        Pattern::Binary { lhs, op, rhs }
    }

}

// TODO: pow of number
/// base^exp
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Pow {
    pub(crate) base: Base,
    pub(crate) exp: Base,
}

impl Pow {
    pub fn pow(n1: impl CalcursType, n2: impl CalcursType) -> Base {
        Self {
            base: n1.base(),
            exp: n2.base(),
        }
        .reduce()
    }

    fn reduce(self) -> Base {
        let b = self.base.desc();
        let e = self.exp.desc();

        if (b.is(Item::Undef) || e.is(Item::Undef))
            || (b.is(Item::Zero) && e.is(Item::Zero))
            || (b.is(Item::Zero) && e.is(Item::Neg))
        {
            // 0^0 / 0^-n => undef
            Undefined.base()
        } else if b.is(Item::One) || e.is(Item::One) {
            // 1^x => 1
            // x^1 => x
            self.base
        } else if !b.is(Item::Zero) && e.is(Item::Zero) {
            // x^0 if x != 0 => 1
            Rational::one().base()
        } else if b.is(Item::Zero) && e.is(Item::Pos) {
            // 0^x if x > 0 => 0
            Rational::zero().base()
        } else if b.is(Item::Numeric) && e.is(Item::PosInf) {
            // x^(+oo) = +oo
            Infinity::pos().base()
        } else if b.is(Item::Numeric) && e.is(Item::NegInf) {
            // x^(-oo) = 0
            Rational::zero().base()
        } else if b.is(Item::Rational) && e.is(Item::MinusOne) {
            // n^-1 => 1 / n
            let r = get_itm!(Rational: self.base);
            (Rational::one() / r).base()
        } else if b.is(Item::Numeric) && e.is(Item::Numeric) {
            let n1 = get_itm!(Numeric: self.base);
            let n2 = get_itm!(Numeric: self.exp);
            n1.checked_pow_num(n2).map(|n| n.base()).unwrap_or(
                Pow {
                    base: n1.base(),
                    exp: n2.base(),
                }
                .base(),
            )
        } else {
            self.base()
        }
    }

    #[inline]
    pub fn desc(&self) -> Pattern {
        let op = Item::Pow;
        let lhs = self.base.desc().to_item();
        let rhs = self.exp.desc().to_item();

        Pattern::Binary { lhs, op, rhs }
    }

    fn fmt_parts(base: &Base, exp: &Base, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let b = base.desc();
        let e = exp.desc();

        if b.is(Item::Atom) && e.is(Item::MinusOne) {
            return write!(f, "1/{}", base);
        }

        if b.is(Item::Atom) && (!b.is(Item::Numeric) || b.is(Item::PosInt)) {
            write!(f, "{base}")?;
        } else {
            write!(f, "({base})")?;
        }

        if e.is(Item::One) {
            Ok(())
        } else if e.is(Item::Atom) && (!e.is(Item::Numeric) || e.is(Item::PosInt)) {
            write!(f, "^{exp}")
        } else {
            write!(f, "^({exp})")
        }
    }
}

impl From<Base> for Pow {
    fn from(value: Base) -> Self {
        if let Base::Pow(pow) = value {
            *pow
        } else {
            Pow {
                base: value,
                exp: Rational::one().base(),
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
enum SumElem {
    Product(Product),
    Atom(Base),
}

impl SumElem {
    fn desc(&self) -> Pattern {
        match self {
            SumElem::Product(p) => p.desc(),
            SumElem::Atom(a) => a.desc(),
        }
    }
}

/// helper container for [Add]
///
///  n1 * mul_1 + n2 * mul_2 + n3 * mul_3 + ... \
/// <=> n1 * {pow_1_1 * pow_1_2 * ... } + n2 * { pow_2_1 * pow_2_2 * ...} + ...
#[derive(Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Default)]
pub(crate) struct Sum {
    elems: Vec<(Numeric, SumElem)>,
    //__args: BTreeMap<Product, Numeric>,
}

impl Sum {

    fn find<'a>(&'a mut self, elem: &SumElem) -> Option<(usize, &'a mut Numeric)> {
        if let Some(indx) = self.elems.iter().position(|e| &e.1 == elem) {
            let coeff = self.elems.get_mut(indx).unwrap();
            Some((indx, &mut coeff.0))
        } else {
            None
        }
    }

    pub fn add(&mut self, itm: Base) {
        match itm {
            Base::Mul(m) => {
                let elem = SumElem::Product(m.product);
                if let Some((indx, coeff)) = self.find(&elem) {
                    *coeff += m.coeff;

                    if coeff.desc().is(Item::Zero) {
                        self.elems.remove(indx);
                    }
                } else {
                    self.elems.push((m.coeff, elem));
                }
            },
            a => {
                let elem = SumElem::Atom(a);
                let one = Rational::one().num();
                if let Some((indx, coeff)) = self.find(&elem) {
                    *coeff += one;

                    if coeff.desc().is(Item::Zero) {
                        self.elems.remove(indx);
                    }
                } else {
                    self.elems.push((one, elem));
                }

            },
        }
    }

    #[inline]
    pub fn into_mul_iter(self) -> impl Iterator<Item = Mul> {
        self.elems
            .into_iter()
            .map(|(coeff, sum_elem)| Mul { coeff, product: match sum_elem {
                SumElem::Product(p) => p,
                SumElem::Atom(a) => a.into(),
            } })
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.elems.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.elems.is_empty()
    }

    #[inline]
    pub fn is_product(&self) -> bool {
        self.len() == 1
    }

    pub fn desc(&self) -> Pattern {
        if self.elems.len() == 1 {
            let (coeff, elem) = self.elems.first().unwrap();
            let op = Item::Mul;
            let lhs = coeff.desc().to_item();
            let rhs = elem.desc().to_item();
            Pattern::Binary { lhs, op, rhs }
        } else {
            Pattern::Itm(Item::Add)
        }
    }
}

impl TryFrom<Sum> for Mul {
    type Error = &'static str;

    fn try_from(mut s: Sum) -> Result<Self, Self::Error> {
        if s.is_product() {
            let (coeff, elem) = s.elems.pop().unwrap();
            match elem {
                SumElem::Product(p) => {
                    Ok(Mul {
                        coeff,
                        product: p
                    })
                },
                SumElem::Atom(a) => {
                    let mut m = Mul::zero();
                    m.coeff = coeff;
                    m.product = a.into();
                    Ok(m)
                },
            }
        } else {
            Err("conversion failed: sum is not a product")
        }
    }
}

/// helper container for [Mul]
///
/// k1 ^ v1 * k2 ^ v2 * k3 ^ v3 * ...
#[derive(Default, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) struct Product {
    elems: Vec<Pow>,
    //__args: BTreeMap<Base, Base>,
}

impl Product {

    fn find(&mut self, elem: &mut Base) -> Option<(usize, &mut Pow)> {
        if let Some(indx) = self.elems.iter().position(|e| &e.base == elem) {
            let coeff = self.elems.get_mut(indx).unwrap();
            Some((indx, coeff))
        } else {
            None
        }
    }

    pub fn zero() -> Self {
        Product {
            elems: vec![],
        }
    }

    pub fn mul(&mut self, itm: Base) {
        let mut pow: Pow = itm.into();
        if let Some((indx, p)) = self.find(&mut pow.base) {
            p.exp += pow.exp;

            if p.exp.desc().is(Item::Zero) {
                self.elems.remove(indx);
            }
        } else {
            self.elems.push(pow);
        }
    }

    #[inline]
    pub fn into_pow_iter(self) -> impl Iterator<Item = Pow> {
        self.elems.into_iter()
    }

    #[inline]
    pub fn is_pow(&self) -> bool {
        self.elems.len() == 1
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.elems.is_empty()
    }

    #[inline]
    pub fn desc(&self) -> Pattern {
        if self.is_pow() {
            self.elems.first().unwrap().desc()
        } else {
            Pattern::Itm(Item::Mul)
        }
    }
}

impl TryInto<Pow> for Product {
    type Error = &'static str;

    fn try_into(mut self) -> Result<Pow, Self::Error> {
        if self.is_pow() {
            Ok(self.elems.pop().unwrap())
        } else {
            Err("product is not a power")
        }
    }
}

impl From<Base> for Product {
    fn from(value: Base) -> Self {
        let mut p = Product::zero();
        p.elems.push(value.into());
        p
    }
}

impl CalcursType for Add {
    #[inline(always)]
    fn base(self) -> Base {
        Base::Add(self)
    }
}

impl CalcursType for Mul {
    #[inline(always)]
    fn base(self) -> Base {
        Base::Mul(self)
    }
}

impl CalcursType for Pow {
    #[inline(always)]
    fn base(self) -> Base {
        Base::Pow(self.into())
    }
}

impl fmt::Display for Add {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.sum)?;

        if !self.coeff.desc().is(Item::Zero) {
            write!(f, " + {}", self.coeff)?;
        }

        Ok(())
    }
}

impl fmt::Display for Mul {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        Mul::fmt_coeff(&self.coeff, self.product.desc(), f)?;
        write!(f, "{}", self.product)
    }
}

impl fmt::Display for Pow {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        Pow::fmt_parts(&self.base, &self.exp, f)
    }
}

impl fmt::Display for Sum {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut iter = self.elems.iter().rev();

        if let Some((coeff, prod)) = iter.next() {
            Mul::fmt_parts(coeff, prod, f)?;
        }

        for (coeff, prod) in iter {
            write!(f, " + ")?;
            Mul::fmt_parts(coeff, prod, f)?;
        }

        Ok(())
    }
}

impl fmt::Display for SumElem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SumElem::Product(p) => write!(f, "{p}"),
            SumElem::Atom(a) => write!(f, "{a}"),
        }
    }
}

impl fmt::Display for Product {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut iter = self.elems.iter();

        if let Some(pow) = iter.next() {
            write!(f, "{pow}")?;
        }

        for pow in iter {
            if pow.exp.desc().is(Item::MinusOne) {
                write!(f, "/{}", pow.base)?;
            } else {
                write!(f, " * {pow}")?;
            }
        }

        Ok(())
    }
}

impl fmt::Debug for Add {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Sum( ")?;
        write!(f, "{:?}", self.sum)?;
        write!(f, " + {:?}", self.coeff)?;
        write!(f, " )")
    }
}

impl fmt::Debug for Mul {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Prod( ")?;
        write!(f, "{:?}", self.product)?;
        write!(f, " * {:?}", self.coeff)?;
        write!(f, " )")
    }
}

impl fmt::Debug for Sum {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut first = true;
        for a in self.elems.iter() {
            if !first {
                write!(f, " + ")?;
            } else {
                first = false
            }
            write!(f, "{:?} * {:?}", a.1, a.0)?;
        }

        Ok(())
    }
}

impl fmt::Debug for Product {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut first = true;
        for a in self.elems.iter() {
            if !first {
                write!(f, " * ")?;
            } else {
                first = false
            }
            write!(f, "{:?}", a)?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod deriv_test {
    use crate::base;
    use crate::prelude::*;
    use pretty_assertions::assert_eq;
    use test_case::test_case;

    #[test_case(1, base!(v: x).pow(base!(2)) + base!(v: x) * base!(3), base!(2) * base!(v: x) + base!(3))]
    #[test_case(2, base!(1 / 3) + base!(3 /  5), base!(0))]
    #[test_case(3, base!(v: x) + base!(v: y), base!(1))]
    fn sum_rule(_case: u32, f: Base, df: Base) {
        assert_eq!(f.derive("x"), df);
    }

    //#[test_case(1, base!(v: x).pow(base!(2)) * base!(v: y))]
    //fn product_rule(_caes: u32, f: Base, df: Base) {
    //    todo!()
    //}
}

#[cfg(test)]
mod op_test {
    use crate::base;
    use crate::prelude::*;
    use pretty_assertions::assert_eq;
    use test_case::test_case;

    #[test_case(1, base!(2), base!(3), base!(5))]
    #[test_case(2, base!(1 / 2), base!(1 / 2), base!(1))]
    #[test_case(3, base!(v: x), base!(v: x), base!(v: x) * base!(2))]
    #[test_case(4, base!(-3), base!(1 / 2), base!(-5 / 2))]
    #[test_case(5, base!(inf), base!(4), base!(inf))]
    #[test_case(6, base!(neg_inf), base!(4), base!(neg_inf))]
    #[test_case(7, base!(pos_inf), base!(pos_inf), base!(pos_inf))]
    #[test_case(8, base!(neg_inf), base!(pos_inf), base!(nan))]
    #[test_case(9, base!(nan), base!(pos_inf), base!(nan))]
    #[test_case(10, base!(4 / 2), base!(0), base!(2))]
    fn add(_case: u32, x: Base, y: Base, z: Base) {
        let expr = x + y;
        assert_eq!(expr, z);
    }

    #[test_case(1, base!(-1), base!(3), base!(-4))]
    #[test_case(2, base!(-3), base!(1 / 2), base!(-7 / 2))]
    #[test_case(3, base!(1 / 2), base!(1 / 2), base!(0))]
    #[test_case(4, base!(inf), base!(4), base!(inf))]
    #[test_case(5, base!(neg_inf), base!(4 / 2), base!(neg_inf))]
    #[test_case(6, base!(pos_inf), base!(4), base!(pos_inf))]
    #[test_case(7, base!(pos_inf), base!(pos_inf), base!(nan))]
    #[test_case(8, base!(neg_inf), base!(pos_inf), base!(neg_inf))]
    #[test_case(9, base!(nan), base!(inf), base!(nan))]
    fn sub(_case: u32, x: Base, y: Base, z: Base) {
        let expr = x - y;
        assert_eq!(expr, z)
    }

    #[test_case(1, base!(-1), base!(3), base!(-3))]
    #[test_case(2, base!(-1), base!(0), base!(0))]
    #[test_case(3, base!(-1), base!(3) * base!(0), base!(0))]
    #[test_case(4, base!(-3), base!(1 / 2), base!(-3 / 2))]
    #[test_case(5, base!(1 / 2), base!(1 / 2), base!(1 / 4))]
    #[test_case(6, base!(inf), base!(4), base!(inf))]
    #[test_case(7, base!(neg_inf), base!(4 / 2), base!(neg_inf))]
    #[test_case(8, base!(pos_inf), base!(4), base!(pos_inf))]
    #[test_case(9, base!(pos_inf), base!(-1), base!(neg_inf))]
    #[test_case(10, base!(pos_inf), base!(pos_inf), base!(pos_inf))]
    #[test_case(11, base!(neg_inf), base!(pos_inf), base!(neg_inf))]
    #[test_case(12, base!(nan), base!(inf), base!(nan))]
    fn mul(_case: u32, x: Base, y: Base, z: Base) {
        let expr = x * y;
        assert_eq!(expr, z);
    }

    #[test_case(1, base!(0), base!(0), base!(nan))]
    #[test_case(2, base!(0), base!(5), base!(0))]
    #[test_case(3, base!(5), base!(0), base!(nan))]
    #[test_case(4, base!(5), base!(5), base!(1))]
    #[test_case(5, base!(1), base!(3), base!(1 / 3))]
    #[test_case(6, base!(v: x), base!(v: x), base!(1))]
    fn div(_case: u32, x: Base, y: Base, z: Base) {
        let div = x / y;
        assert_eq!(div, z);
    }

    #[test_case(1, base!(1), base!(1 / 100), base!(1))]
    #[test_case(2, base!(4), base!(1), base!(4))]
    #[test_case(3, base!(0), base!(0), base!(nan))]
    #[test_case(4, base!(0), base!(-3 / 4), base!(nan))]
    #[test_case(5, base!(0), base!(3 / 4), base!(0))]
    #[test_case(6, base!(1 / 2), base!(-1), base!(4 / 2))]
    fn pow(_case: u32, x: Base, y: Base, z: Base) {
        let expr = x.pow(y);
        assert_eq!(expr, z);
    }

    #[test]
    fn polynom() {
        let p1 = base!(v: x) * base!(v: x) * base!(2) + base!(3) * base!(v: x) + base!(4 / 3);
        let p2 = base!(4 / 3) + (base!(v: x).pow(base!(2))) * base!(2) + base!(3) * base!(v: x);
        assert_eq!(p1, p2);
    }
}
