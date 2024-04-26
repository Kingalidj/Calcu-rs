use std::fmt;

use crate::{
    base::{Base, CalcursType},
    numeric::Numeric,
    pattern::{get_itm, Item, Pattern},
    rational::Rational,
};

use calcu_rs::{calc, identity};

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

    /// n1 + (-1 * n2)
    #[inline]
    pub fn sub(n1: impl CalcursType, n2: impl CalcursType) -> Base {
        Add::add(n1, Mul::mul(Rational::minus_one(), n2))
    }

    pub fn arg(&mut self, b: Base) {
        use Base as B;
        match b {
            B::Rational(r) => self.coeff += Numeric::Rational(r),
            B::Infinity(i) => self.coeff += Numeric::Infinity(i),
            B::Float(f) => self.coeff += Numeric::Float(f),
            B::Undefined => self.coeff += Numeric::Undefined,
            B::Mul(mul) => self.sum.add(mul.base()),
            B::Add(add) => {
                self.coeff += add.coeff;
                add.sum
                    .into_mul_iter()
                    .for_each(|mul| self.sum.add(mul.base()));
            }

            base @ (B::Symbol(_) | B::Pow(_)) => self.sum.add(base),
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
impl CalcursType for Add {
    fn desc(&self) -> Pattern {
        let op = Item::Add;
        let lhs = self.coeff.desc().get_item();
        let rhs = self.sum.desc().get_item();
        Pattern::Binary { lhs, op, rhs }
    }

    #[inline(always)]
    fn base(self) -> Base {
        Base::Add(self)
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
        println!("mul: [{}] * [{}]", n1.clone().base(), n2.clone().base());
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

        if b == Numeric::Undefined.base() {
            self.coeff = Numeric::Undefined;
            return;
        } else if self.coeff == Rational::zero().num() || Rational::one().base() == b {
            return;
        }

        match b {
            B::Rational(r) => self.coeff *= Numeric::Rational(r),
            B::Infinity(i) => self.coeff *= Numeric::Infinity(i),
            B::Float(f) => self.coeff *= Numeric::Float(f),
            B::Undefined => self.coeff *= Numeric::Undefined,
            B::Mul(mul) => {
                self.coeff *= mul.coeff;
                mul.product
                    .into_pow_iter()
                    .for_each(|pow| self.arg(pow.base()));
            }
            B::Pow(pow) => {
                if pow.base.desc().is(Item::Atom) {
                    self.product.mul(pow.base());
                } else {
                    let flat = pow.expand();
                    // the base of the power is not an atom, so the flattened version should no
                    // longer be a power
                    debug_assert!(!flat.desc().is(Item::Pow));
                    self.arg(flat);
                }
            }

            //self.product.mul(pow.base()),
            B::Symbol(_) | B::Add(_) => self.product.mul(b),
        }
    }

    pub fn reduce(self) -> Base {
        let coeff = self.coeff.desc();
        let prod = self.product.desc();

        if self.product.is_empty() {
            return self.coeff.base();
        }

        identity! { (coeff, prod) {
            (Item::Undef, _) => calc!(undef),
            (Item::Zero, _) => calc!(0),
            (Item::One, Item::Pow) => {
                let p: Pow = self.product.try_into().unwrap();
                p.base()
            },
            default => self.base(),
        }}
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
        let p = prod.desc();
        Self::fmt_coeff(coeff, p, f)?;
        if p.is(Item::Pow) || p.is(Item::Atom) {
            write!(f, "{prod}")
        } else {
            write!(f, "({prod})")
        }
    }
}
impl CalcursType for Mul {
    #[inline]
    fn desc(&self) -> Pattern {
        let op = Item::Mul;
        let lhs = self.coeff.desc().get_item();
        let rhs = self.product.desc().get_item();
        Pattern::Binary { lhs, op, rhs }
    }

    #[inline(always)]
    fn base(self) -> Base {
        Base::Mul(self)
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
        use Item as I;

        let b = self.base.desc();
        let e = self.exp.desc();

        identity! { (b, e) {
            // undef => undef
            (I::Undef, _) || (_, I::Undef)
                // 0^0 / 0^-n => undef
                || (I::Zero, I::Zero) || (I::Zero, I::Neg) => calc!(undef),

                // 1^x => 1
                // x^1 => x
                (_, I::One) || (I::One, _) => self.base,
                // x^0 => 1 | if x != 0
                (!I::Zero, I::Zero) => calc!(1),

                // 0^x => 0 | if x > 0
                (I::Zero, I::Pos) => calc!(0),

                // n^(-1) => 1 / n
                (I::Rational, I::MinusOne) => {
                    // from div?
                    let r = get_itm!(Rational: self.base);
                    //TODO: inv()
                    Rational::one().convert_div(r).base()
                },

                // (x^a)^b => x^(a*b) | if b in Z
                (I::Pow, I::Int) => {
                    let mut pow = get_itm!(Pow: self.base);
                    pow.exp *= self.exp;
                    pow.base()
                },

                (I::Numeric, I::Numeric) => {
                    let n1: Numeric = self.base.try_into().expect("Numerick");
                    let n2: Numeric = self.exp.try_into().expect("Numerick");
                    n1.checked_pow_num(n2).map(|n| n.base()).unwrap_or(
                        Pow {
                            base: n1.base(),
                            exp: n2.base(),
                        }
                        .base(),
                        )
                },

                //TODO: x^(+oo) = +oo

                default => self.base(),
        }}
    }

    pub fn expand(self) -> Base {
        use Item as I;

        let b = self.base.desc();
        let e = self.exp.desc();

        identity! {(b, e) {

            (I::Add, I::Int) => {
                let add = get_itm!(Add: self.base);
                let rat = get_itm!(Rational: self.exp);
                let int = rat.numer;

                match int {
                    1 => add.base(),
                    2 => {
                        todo!()
                    }
                    _ => add.base().pow(rat),
                }
            },

            // (a * b * c)^d => a^d * b^d * c^d
            // powers with same exponent will not be reduced
            (I::Mul, _) => {
                let mul = get_itm!(Mul: self.base);
                let mul_coeff = mul.coeff;
                let mul_elems = mul.product.elems;

                let mut flat_res = mul_coeff.base().pow(self.exp.clone());

                for elem in mul_elems {
                    flat_res *= elem.base().pow(self.exp.clone());
                }

                flat_res
            },

            default => self.base(),
        }}
    }

    fn fmt_parts(base: &Base, exp: &Base, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let b = base.desc();
        let e = exp.desc();

        if b.is(Item::Atom) && e.is(Item::MinusOne) {
            return write!(f, "1/{base}");
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
impl CalcursType for Pow {
    #[inline]
    fn desc(&self) -> Pattern {
        let op = Item::Pow;
        let lhs = self.base.desc().get_item();
        let rhs = self.exp.desc().get_item();

        Pattern::Binary { lhs, op, rhs }
    }

    #[inline(always)]
    fn base(self) -> Base {
        Base::Pow(self.into())
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
}

impl Sum {
    fn find(&mut self, elem: &SumElem) -> Option<(usize, &mut Numeric)> {
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
            }
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
            }
        }
    }

    #[inline]
    pub fn into_mul_iter(self) -> impl Iterator<Item = Mul> {
        self.elems.into_iter().map(|(coeff, sum_elem)| Mul {
            coeff,
            product: match sum_elem {
                SumElem::Product(p) => p,
                SumElem::Atom(a) => a.into(),
            },
        })
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
            let lhs = coeff.desc().get_item();
            let rhs = elem.desc().get_item();
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
                SumElem::Product(p) => Ok(Mul { coeff, product: p }),
                SumElem::Atom(a) => {
                    let mut m = Mul::zero();
                    m.coeff = coeff;
                    m.product = a.into();
                    Ok(m)
                }
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
}

impl Product {
    // TODO: find all terms that contain this base
    //TODO: rewrite
    fn find_base(&mut self, elem: &Base) -> Option<(usize, &mut Pow)> {
        // search for expressions that can be shortened:
        if let Some(indx) = self.elems.iter().position(|e| //&e.base == elem
            {
                match elem {
                    Base::Pow(pow) =>
                        {
                            println!("{}: find_base: {}", e, pow);
                            pow.base == e.base
                        },
                    Base::Rational(_) | Base::Float(_) | Base::Infinity(_) | Base::Undefined => false,
                    Base::Mul(_) => panic!("multiplication should happen element-wise"),
                    _ => elem == &e.base,
                }
            })
        {
            let coeff = self.elems.get_mut(indx).unwrap();
            Some((indx, coeff))
        } else {
            None
        }
    }

    pub fn zero() -> Self {
        Product { elems: vec![] }
    }

    pub fn mul(&mut self, itm: Base) {
        if let Some((indx, p)) = self.find_base(&itm) {
            let pow: Pow = itm.into();
            p.exp += pow.exp;

            if p.exp.desc().is(Item::Zero) {
                self.elems.remove(indx);
            }
        } else {
            self.elems.push(itm.into());
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
