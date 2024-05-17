use crate::expression::Expr;
use crate::expression::CalcursType;
use crate::pattern::Item;
use crate::rational::Rational;
use std::fmt;
use std::fmt::Formatter;

pub type OperandSet = Vec<Expr>;

#[derive(Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Sum {
    pub operands: OperandSet,
}
pub type Diff = Sum;

impl Sum {
    #[inline]
    pub fn sum(lhs: impl CalcursType, rhs: impl CalcursType) -> Expr {
        let lhs = lhs.into();
        let rhs = rhs.into();
        match (lhs, rhs) {
            (Expr::Undefined, _) | (_, Expr::Undefined) => Expr::Undefined,
            (Expr::ZERO, other) | (other, Expr::ZERO) => other,
            (lhs, rhs) => {
                Self::zero().arg(lhs).arg(rhs).into()
            }
        }
    }

    #[inline]
    pub fn diff(lhs: impl CalcursType, rhs: impl CalcursType) -> Expr {
        let lhs = lhs.into();
        let rhs = Expr::MINUS_ONE * rhs.into();
        Self::sum(lhs, rhs)
    }

    #[inline]
    pub fn sum_raw(lhs: impl CalcursType, rhs: impl CalcursType) -> Expr {
        let mut sum = Self::zero();
        sum.operands.push(lhs.into());
        sum.operands.push(rhs.into());
        sum.into()
    }
    #[inline]
    pub fn diff_raw(lhs: impl CalcursType, rhs: impl CalcursType) -> Expr {
        let lhs = lhs.into();
        let rhs = Expr::MINUS_ONE * rhs.into();
        Self::sum_raw(lhs, rhs)
    }

    #[inline]
    fn arg(mut self, b: Expr) -> Self {
        use Expr as E;
        match b {
            E::Sum(mut add) => self.operands.append(&mut add.operands),
            _ => self.operands.push(b),
        }
        self
    }

    pub fn zero() -> Self {
        Self {
            operands: Default::default(),
        }
    }
}

impl CalcursType for Sum {
    fn desc(&self) -> Item {
        Item::Sum
    }
}


#[derive(Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Prod {
    pub operands: OperandSet,
}
pub type Quot = Prod;

impl Prod {
    #[inline]
    pub fn prod(lhs: impl CalcursType, rhs: impl CalcursType) -> Expr {
        let lhs = lhs.into();
        let rhs = rhs.into();

        match (lhs, rhs) {
            (Expr::Undefined, _) | (_, Expr::Undefined) => Expr::Undefined,
            (Expr::ZERO, _) | (_, Expr::ZERO) => Expr::ZERO,
            (Expr::ONE, other) | (other, Expr::ONE) => other,
            (Expr::Rational(r1), Expr::Rational(r2)) => (r1 * r2).into(),
            (lhs, rhs) => Self::zero().arg(lhs).arg(rhs).into()
        }
    }

    #[inline]
    pub fn quot(lhs: impl CalcursType, rhs: impl CalcursType) -> Expr {
        let lhs = lhs.into();
        let rhs = rhs.into().pow(Expr::MINUS_ONE);
        Self::prod(lhs, rhs)
    }

    #[inline]
    pub fn prod_raw(lhs: impl CalcursType, rhs: impl CalcursType) -> Expr {
        let mut prod = Self::zero();
        prod.operands.push(lhs.into());
        prod.operands.push(rhs.into());
        prod.into()
    }
    #[inline]
    pub fn quot_raw(lhs: impl CalcursType, rhs: impl CalcursType) -> Expr {
        let lhs = lhs.into();
        let rhs = rhs.into().pow(Expr::MINUS_ONE);
        Self::prod_raw(lhs, rhs)
    }

    fn arg(mut self, b: Expr) -> Self {
        match b {
            Expr::Prod(mut mul) => self.operands.append(&mut mul.operands),
            _ => self.operands.push(b),
        }
        self
    }

    fn zero() -> Self {
        Self {
            operands: Default::default(),
        }
    }
}

impl CalcursType for Prod {
    fn desc(&self) -> Item {
        Item::Prod
    }
}

#[derive(Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
#[repr(C)]
pub struct Pow {
    pub(crate) operands: [Expr; 2], // [base, exponent]
}

impl Pow {
    #[inline(always)]
    pub fn base(&self) -> &Expr {
        unsafe { self.operands.get_unchecked(0) }
    }

    #[inline(always)]
    pub fn exponent(&self) -> &Expr {
        unsafe { self.operands.get_unchecked(1) }
    }

    #[inline]
    pub fn pow(b: impl CalcursType, e: impl CalcursType) -> Expr {
        let base = b.into();
        let exp = e.into();

        match (base, exp) {
            (Expr::Undefined, _) | (_,  Expr::Undefined) => Expr::Undefined,

            (Expr::ZERO,  Expr::ZERO) =>  Expr::Undefined,
            (Expr::ZERO,  Expr::Rational(exp)) if exp.is_neg() => Expr::Undefined,
            (Expr::ZERO,  Expr::Rational(exp)) if exp.is_pos() => Expr::ZERO,

            ( Expr::ONE, _) =>  Expr::ONE,
            (b,  Expr::ONE) => b,

            ( Expr::Rational(b),  Expr::Rational(e)) => {
                // when the exponent is non-int:
                // we apply the exponent floor(e1 / e2), so the remaining
                // exponent is < 1
                // b^e => b^(e1 / e2) => b^quot * b^rem
                let float_exp = e.is_int();
                // pow = b^pow, rem = rem in b^rem
                let (pow, rem) = b.clone().pow(e);
                if rem.is_zero() {
                    pow.into()
                } else {
                    let lhs = Expr::from(pow);
                    let rhs = Pow { operands: [b.into(), rem.into()] }.into();
                    lhs * rhs
                }
            }
            (base, exponent) =>  Expr::Pow(
                Pow { operands: [base, exponent] }.into(),
            ),
        }
    }

    #[inline]
    pub fn pow_raw(base: impl CalcursType, exp: impl CalcursType) -> Expr {
        Self { operands: [base.into(), exp.into()] }.into()
    }
}

impl CalcursType for Pow {
    fn desc(&self) -> Item {
        Item::Pow
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
        write!(f, "{}^({})", self.base(), self.exponent())
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
