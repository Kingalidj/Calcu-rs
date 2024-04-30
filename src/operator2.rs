use std::collections::VecDeque;
use crate::expression::Expr;
use calcu_rs::expression::{CalcursType, Construct};
use calcu_rs::pattern::Item;
use calcu_rs::rational::Rational;
use std::fmt;
use std::fmt::Formatter;
use calcurs_macros::identity;

pub type OperandVec = VecDeque<Expr>;

#[derive(Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Sum {
    operands: OperandVec,
}
pub type Diff = Sum;

impl Sum {
    #[inline]
    pub fn add(lhs: impl CalcursType, rhs: impl CalcursType) -> Expr {
        let lhs = lhs.into();
        let rhs = rhs.into();
        Self::zero().arg(lhs).arg(rhs).into()
    }

    #[inline]
    pub fn sub(lhs: impl CalcursType, rhs: impl CalcursType) -> Expr {
        let lhs = lhs.into();
        let rhs = Prod::mul(rhs.into(), Rational::MINUS_ONE);
        Self::add(lhs, rhs)
    }

    fn arg(mut self, b: Expr) -> Self {
        use Expr as E;
        match b {
            E::Sum(mut add) => self.operands.append(&mut add.operands),
            _ => self.operands.push_back(b),
        }
        self
    }

    pub fn zero() -> Self {
        Self { operands: Default::default() }
    }

    fn merge_sums(s1: &[Expr], s2: &[Expr]) -> OperandVec {
        todo!()
    }

    /// helper function for [Self::simplify_rec]
    /// used in the case of u1 + u2
    fn simplify_rec_2(u1: Expr, u2: Expr) -> OperandVec {
        use Expr as E;
        let (d1, d2) = (u1.desc(), u2.desc());

        let mut out = OperandVec::new();

        if d1.is(Item::Sum) || d2.is(Item::Sum) {
            match (u1, u2) {
                (E::Sum(mut s1), E::Sum(mut s2)) => {
                    out = Self::merge_sums(s1.operands.make_contiguous(), s2.operands.make_contiguous());
                }
                (E::Sum(mut s), u2) => {
                    out = Self::merge_sums(s.operands.make_contiguous(), &[u2]);
                }
                (u1, E::Sum(mut s)) => {
                    out = Self::merge_sums(&[u1], s.operands.make_contiguous());
                }
                _ => unreachable!("u1, u2 must be sums"),
            }

        } else if d1.is(Item::Constant) && d2.is(Item::Constant) {
            let s = Sum::add(u1, u2).simplify();

            if s.desc().is_not(Item::Zero) {
                out.push_back(s);
            }
        } else if d1.is(Item::Zero) {
            out.push_back(u2);
        } else if d2.is(Item::Zero) {
            out.push_back(u1);
        } else if u1.base().is_some() && u1.base() == u2.base() {
            // n*b + m*b = (n + m) * b
            let u = u1.base().unwrap().clone();
            todo!()
        } else if u2 < u1 {
            out.push_back(u2);
            out.push_back(u1);
        } else {
            out.push_back(u1);
            out.push_back(u2);
        }

        out
    }

    fn simplify_rec(operands: &mut OperandVec) {
        if operands.len() < 2 {
            return;
        }

        if operands.len() == 2 {
            let u1 = operands.pop_front().unwrap();
            let u2 = operands.pop_front().unwrap();
            *operands = Self::simplify_rec_2(u1, u2);
        } else {
            let u1 = operands.pop_front().unwrap().clone();
            Self::simplify_rec(operands);

            if let Expr::Sum(mut s) = u1 {
                *operands = Self::merge_sums(s.operands.make_contiguous(), operands.make_contiguous())
            } else {
                *operands = Self::merge_sums(&[u1], operands.make_contiguous());
            }
        }
    }
}

impl CalcursType for Sum {
    fn desc(&self) -> Item {
        Item::Sum
    }
}

impl Construct for Sum {
    fn free_of(&self, other: &Expr) -> bool {
        if let Expr::Sum(add) = other {
            if self == add {
                return false;
            }
        }

        for op in &self.operands {
            if !op.free_of(other) {
                return false;
            }
        }
        true
    }

    #[inline]
    fn operands(&mut self) -> Vec<&mut Expr> {
        self.operands.iter_mut().collect()
    }

    fn simplify(mut self) -> Expr {
        if self.operands.len() == 1 {
            return self.operands.pop_front().unwrap();
        }

        for op in &self.operands {
            let d = op.desc();

            if d.is(Item::Undef) {
                return Expr::Undefined;
            }
        }

        Self::simplify_rec(&mut self.operands);

        if self.operands.is_empty() {
            Rational::ZERO
        } else if self.operands.len() == 1 {
            self.operands.pop_front().unwrap()
        } else {
            Expr::Sum(self)
        }
    }

    fn all_variables(&self) -> Vec<Expr> {
        let mut vars = vec![];
        for op in &self.operands {
            vars.append(&mut op.all_variables());
        }
        vars
    }
}

#[derive(Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Prod {
    operands: OperandVec,
}
pub type Quot = Prod;

impl Prod {
    pub fn mul(lhs: impl CalcursType, rhs: impl CalcursType) -> Expr {
        let lhs = lhs.into();
        let rhs = rhs.into();
        Self::zero().arg(lhs).arg(rhs).into()
    }

    pub fn div(lhs: impl CalcursType, rhs: impl CalcursType) -> Expr {
        let lhs = lhs.into();
        let rhs = Pow::pow(rhs.into(), Rational::MINUS_ONE);
        Self::mul(lhs, rhs)
    }

    fn arg(mut self, b: Expr) -> Self {
        match b {
            Expr::Prod(mut mul) => self.operands.append(&mut mul.operands),
            _ => self.operands.push_back(b),
        }

        self
    }

    fn zero() -> Self {
        Self { operands: Default::default() }
    }

    /// merges two operand slices
    fn merge_prods(p: &[Expr], q: &[Expr]) -> OperandVec {
        if q.is_empty() {
            p.iter().cloned().collect()
        } else if p.is_empty() {
            q.iter().cloned().collect()
        } else {
            let p1 = p.get(0).unwrap();
            let q1 = q.get(0).unwrap();

            let mut h = OperandVec::from([p1.clone(), q1.clone()]);
            Self::simplify_rec(&mut h);

            if h.is_empty() {
                // p1, q1 cancel out
                return Self::merge_prods(&p[1..], &q[1..]);
            } else if h.len() == 1 {
                // p1, q1 merged into h1
                h.append(&mut Self::merge_prods(&p[1..], &q[1..]));
                h
            } else if h.len() == 2 {
                // p1, q1 could not be merged
                if h.get(0) == Some(p1) {
                    // h = [p1, q1]
                    let mut merged = Self::merge_prods(&p[1..], q);
                    merged.push_front(p1.clone());
                    merged

                } else {
                    // h = [q1, p1]
                    debug_assert_eq!(h.get(0), Some(q1));
                    let mut merged = Self::merge_prods(p, &q[1..]);
                    merged.push_front(q1.clone());
                    merged
                }
            } else {
                unreachable!("two elements simplified should not become more elements")
            }
        }
    }

    /// helper function for [Self::simplify_rec]
    /// used in the case of u1 * u2
    #[inline(always)]
    fn simplify_rec_2(u1: Expr, u2: Expr) -> OperandVec {
        use Expr as E;
        let (d1, d2) = (u1.desc(), u2.desc());

        let mut out = OperandVec::default();

        if d1.is(Item::Prod) || d2.is(Item::Prod) {
            // merge into existing product
            match (u1, u2) {
                (E::Prod(mut p), E::Prod(mut q)) => {
                    out = Self::merge_prods(p.operands.make_contiguous(), q.operands.make_contiguous())
                }
                (E::Prod(mut p), u2) => {
                    out = Self::merge_prods(p.operands.make_contiguous(), &[u2]);
                }
                (u1, E::Prod(mut p)) => {
                    out = Self::merge_prods(&[u1], p.operands.make_contiguous());
                }
                _ => unreachable!("u1, u2 must be products")
            }
        } else if d1.is(Item::Constant) && d2.is(Item::Constant) {
            // c1*c2, where c1, c2 are constants
            let p = Prod::mul(u1, u2).simplify(); //TODO: check non recursive
            if p.desc().is_not(Item::One) {
                out.push_back(p);
            }
        } else if d1.is(Item::One) {
            // 1 * u2 = u2
            out.push_back(u2);
        } else if d2.is(Item::One) {
            // u1 * 1 = u1
            out.push_back(u1)
        } else if u1.base().is_some() && u1.base() == u2.base() {
            // b^e1 * b^e2 -> b^(e1 + e2)
            let u = u1.base().unwrap().clone();
            let e1 = u1.exponent().unwrap().clone(); // if u1 is Some => e1 must also be Some;
            let e2 = u2.exponent().unwrap().clone();
            let s = Sum::add(e1, e2).simplify();
            let p = Pow::pow(u, s).simplify();

            if p.desc().is_not(Item::One) {
                out.push_back(p);
            }
        } else if u2 < u1 {
            out.push_back(u2);
            out.push_back(u1);
        } else {
            out.push_back(u1);
            out.push_back(u2);
        };

        out
    }

    fn simplify_rec(operands: &mut OperandVec) {
        if operands.len() < 2 {
            return;
        }

        if operands.len() == 2 {
            let u1 = operands.pop_front().unwrap();
            let u2 = operands.pop_front().unwrap();
            *operands = Self::simplify_rec_2(u1, u2);
        } else {
            let u1 = operands.pop_front().unwrap().clone();
            Self::simplify_rec(operands);

            if let Expr::Prod(mut p) = u1 {
                *operands = Self::merge_prods(p.operands.make_contiguous(), operands.make_contiguous())
            } else {
                *operands = Self::merge_prods(&[u1], operands.make_contiguous())
            }
        }
    }
}

impl CalcursType for Prod {
    fn desc(&self) -> Item {
        Item::Prod
    }

}

impl Construct for Prod {
    fn free_of(&self, other: &Expr) -> bool {
        if let Expr::Prod(mul) = other {
            if self == mul {
                return false;
            }
        }
        for op in &self.operands {
            if !op.free_of(other) {
                return false;
            }
        }
        true
    }

    #[inline]
    fn operands(&mut self) -> Vec<&mut Expr> {
        self.operands.iter_mut().collect()
    }

    fn simplify(mut self) -> Expr {
        if self.operands.len() == 1 {
            return self.operands.pop_front().unwrap();
        }

        // filter out zero and undefined
        for op in &self.operands {
            let d = op.desc();

            if d.is(Item::Zero) {
                return Rational::ZERO;
            } else if d.is(Item::Undef) {
                return Expr::Undefined;
            }
        }

        Self::simplify_rec(&mut self.operands);

        if self.operands.is_empty() {
            Rational::ONE
        } else if self.operands.len() == 1 {
            self.operands.pop_front().unwrap()
        } else {
            Expr::Prod(self)
        }
    }

    fn all_variables(&self) -> Vec<Expr> {
        let mut vars = vec![];
        for op in &self.operands {
            vars.append(&mut op.all_variables());
        }
        vars
    }
}

#[derive(Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
pub struct Pow {
    pub(crate) base: Expr,
    pub(crate) exponent: Expr,
}

impl Pow {
    #[inline(always)]
    pub fn new(b: impl CalcursType, e: impl CalcursType) -> Pow {
        Self {
            base: b.into(),
            exponent: e.into(),
        }
    }
    #[inline]
    pub fn pow(b: impl CalcursType, e: impl CalcursType) -> Expr {
        Expr::Pow(Self::new(b, e).into())
    }

    // x^n where x is an integer
    fn simplify_int_pow(base: Expr, n: i64) -> Expr {
        use Expr as E;

        if n == 0 {
            return Rational::ONE;
        } else if n == 1 {
            return base;
        }

        match base {
            // (r^s)^n = r^(s*n)
            E::Pow(pow) => {
                let r = pow.base;
                let s = pow.exponent;
                let p = Prod::mul(s, Rational::from(n)).simplify();
                Pow::pow(r, p).simplify()
            }
            // v^n = (v1 * ... * vm)^n = v1^n * ... * vm^n
            E::Prod(mut prod) => {
                prod.map(|elem| {
                    *elem = Self::simplify_int_pow(elem.clone(), n);
                });
                prod.simplify()
            }
            _ => E::Pow(Pow::new(base, Rational::from(n)).into())
        }
    }
}

impl CalcursType for Pow {
    fn desc(&self) -> Item {
        Item::Pow
    }
}

impl Construct for Pow {
    fn free_of(&self, other: &Expr) -> bool {
        if let Expr::Pow(pow) = other {
            if self == pow.as_ref() {
                return false;
            }
        } else if self.exponent.desc().is(Item::One) {
            if !self.base.free_of(other) {
                return false;
            }
        }

        true
    }

    #[inline]
    fn operands(&mut self) -> Vec<&mut Expr> {
        vec![&mut self.base, &mut self.exponent]
    }

    fn simplify(self) -> Expr {
        use Expr as E;
        use Item as I;

        let base_desc = self.base.desc();
        let exp_desc = self.exponent.desc();

        // special cases
        identity!((base_desc, exp_desc) {
            // undef -> undef
            (I::Undef, _)
            || (_, I::Undef)
            // 0^0 -> undef
            || (I::Zero, I::Zero)
            // 0^x, x < 0 -> undef
            || (I::Zero, I::Neg) => {
                return E::Undefined;
            },
            // 0^(x), x > 0 -> 1
            (I::Zero, I::Pos) => {
                return Rational::ONE;
            },
            // 1^x -> 1
            (I::One, _) => {
                return Rational::ONE;
            },
            // x^1 -> x
            (_, I::One) => {
                return E::Pow(self.into());
            },

            default => {}
        });

        match (self.base, self.exponent) {
            (E::Rational(r1), E::Rational(r2)) => {
                E::Pow(r1.apply_pow(r2).into())
            }
            (E::Float(f1), E::Float(f2)) => {
                E::Float(f1.pow(f2))
            }
            (E::Float(f), E::Rational(r)) => {
                E::Float(f.pow(r.to_float()))
            }
            (E::Rational(r), E::Float(f)) => {
                E::Float(r.to_float().pow(f))
            }
            // integer power
            (base, E::Rational(n)) if n.is_int() => {
                Self::simplify_int_pow(base, n.to_int())
            }
            (base, exp) => E::Pow(Pow { base, exponent: exp }.into()),
        }
    }

    fn all_variables(&self) -> Vec<Expr> {
        if self.exponent.desc().is(Item::Constant) {
            self.base.all_variables()
        } else {
            //TODO: what here?
            vec![]
        }
    }
}

impl fmt::Display for Sum {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        if self.operands.is_empty() {
            return Ok(());
        }

        let mut iter = self.operands.iter();
        write!(f, "{}", iter.next().unwrap())?;

        for elem in iter {
            write!(f, " + {}", elem)?;
        }

        Ok(())
    }
}

impl fmt::Display for Prod {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        if self.operands.is_empty() {
            return Ok(());
        }

        let mut iter = self.operands.iter();
        write!(f, "{}", iter.next().unwrap())?;

        for elem in iter {
            write!(f, " * {}", elem)?;
        }

        Ok(())
    }
}

impl fmt::Display for Pow {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{}^{}", self.base, self.exponent)
    }
}

impl fmt::Debug for Sum {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "Add[")?;
        let mut iter = self.operands.iter();
        if let Some(e) = iter.next() {
            write!(f, "{:?}", e)?;
        }
        for elem in iter {
            write!(f, ", {:?}", elem)?;
        }
        write!(f, "]")
    }
}

impl fmt::Debug for Prod {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "Mul[ ")?;
        let mut iter = self.operands.iter();
        if let Some(e) = iter.next() {
            write!(f, "{:?}", e)?;
        }
        for elem in iter {
            write!(f, ", {:?}", elem)?;
        }
        write!(f, "]")
    }
}
