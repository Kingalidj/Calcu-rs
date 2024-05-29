use calcu_rs::{
    expression::{CalcursType, Construct, Expr},
    pattern::Item,
    rational::Rational,
};
use calcurs_macros::identity;
use std::fmt::{self, Display, Formatter};

pub type OperandSet = Vec<Expr>;

#[derive(Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Sum {
    pub operands: OperandSet,
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

impl Construct for Sum {
    fn simplify(mut self) -> Expr {
        for op in &mut self.operands {
            let mut e = Expr::Undefined;
            std::mem::swap(&mut e, op);
            *op = e.simplify();

            if let Expr::Undefined = op {
                return Expr::Undefined;
            }
        }

        if self.operands.is_empty() {
            return Rational::ZERO;
        } else if self.operands.len() == 1 {
            return self.operands.pop().unwrap();
        }

        self.operands.sort();
        self.into()
    }
}

#[derive(Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Prod {
    pub operands: OperandSet,
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

impl Construct for Prod {
    fn simplify(mut self) -> Expr {
        for op in &mut self.operands {
            let mut e = Expr::Undefined;
            std::mem::swap(&mut e, op);
            *op = e.simplify();

            if op.desc().is(Item::Zero) {
                return Rational::ZERO;
            } else if let Expr::Undefined = op {
                return Expr::Undefined;
            }
        }

        if self.operands.is_empty() {
            return Rational::ONE;
        } else if self.operands.len() == 1 {
            return self.operands.pop().unwrap();
        }

        self.operands.sort();
        Expr::Prod(self)
    }
}

#[derive(Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
#[repr(C)]
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

    pub fn operands(&self) -> &[Expr] {
        let ptr = unsafe {
            std::slice::from_raw_parts(
                (self as *const Pow) as *const Expr,
                std::mem::size_of::<Self>(),
            )
        };

        assert_eq!(std::mem::size_of::<Self>(), 2 * std::mem::size_of::<Expr>());
        assert_eq!(ptr[0], self.base);
        assert_eq!(ptr[1], self.exponent);
        ptr
    }

    pub fn operands_mut(&mut self) -> &mut [Expr] {
        let ptr = unsafe {
            std::slice::from_raw_parts_mut(
                (self as *mut Pow) as *mut Expr,
                std::mem::size_of::<Self>(),
            )
        };

        assert_eq!(std::mem::size_of::<Self>(), 2 * std::mem::size_of::<Expr>());
        assert_eq!(ptr[0], self.base);
        assert_eq!(ptr[1], self.exponent);
        ptr
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
                prod.operands = prod
                    .operands
                    .into_iter()
                    .map(|e| Self::simplify_int_pow(e, n))
                    .collect();
                prod.simplify()
            }
            _ => E::Pow(Pow::new(base, Rational::from(n)).into()),
        }
    }
}

impl CalcursType for Pow {
    fn desc(&self) -> Item {
        Item::Pow
    }
}

impl Construct for Pow {
    #[inline]
    fn simplify(mut self) -> Expr {
        use Expr as E;
        use Item as I;

        self.base = self.base.simplify();
        self.exponent = self.exponent.simplify();

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
                let r = r1.clone();
                let (pow, rem) = r1.pow(r2);
                Prod::mul(pow, Pow::pow(r, rem))
            }
            //(E::Float(f1), E::Float(f2)) => E::Float(f1.pow(f2)),
            //(E::Float(f), E::Rational(r)) => E::Float(f.pow(r.to_float())),
            //(E::Rational(r), E::Float(f)) => E::Float(r.to_float().pow(f)),
            // integer power
            (base, E::Rational(n)) if n.is_int() => {
                if let Some(int_val) = n.try_into_int() {
                    Self::simplify_int_pow(base, int_val)
                } else {
                    Pow {
                        base,
                        exponent: E::Rational(n),
                    }
                    .into()
                }
            }

            (base, exp) => E::Pow(
                Pow {
                    base,
                    exponent: exp,
                }
                .into(),
            ),
        }
    }
}

impl Display for Sum {
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

impl Display for Prod {
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

impl Display for Pow {
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
