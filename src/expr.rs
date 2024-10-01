use std::{borrow::Borrow, fmt, iter, ops, slice, sync::Arc};

use crate::{
    fmt_ast,
    polynomial::{MonomialView, PolynomialView, VarSet},
    rational::{Int, Rational},
    utils::{self, HashSet},
};

use paste::paste;

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Sum {
    pub(crate) args: Vec<Expr>,
}
impl Sum {
    pub fn zero() -> Self {
        Self {
            args: Default::default(),
        }
    }

    fn is_zero(&self) -> bool {
        self.args.is_empty()
    }
    fn is_undef(&self) -> bool {
        self.args.first().is_some_and(|e| e.is_undef())
    }

    fn first(&self) -> Option<&Atom> {
        self.args.first().map(|a| a.get())
    }

    fn as_binary_sum(&self) -> (Expr, Expr) {
        if self.args.is_empty() {
            (Expr::zero(), Expr::zero())
        } else if self.args.len() == 1 {
            (self.args[0].clone(), Expr::zero())
        } else {
            let a = self.args.first().unwrap().clone();
            let rest = &self.args[1..];
            let b = if rest.len() == 1 {
                rest[0].clone()
            } else {
                Expr::from(Atom::Sum(Sum {
                    args: rest.into_iter().cloned().collect(),
                }))
            };
            (a, b)
        }
    }

    pub fn add_rhs(&mut self, rhs: &Expr) {
        use Atom as A;

        if let Some(A::Undef) = self.first() {
            return;
        }

        match rhs.get() {
            &A::ZERO => (),
            A::Sum(sum) => {
                sum.args.iter().for_each(|a| self.add_rhs(a));
            }
            // sum all rationals in the first element
            A::Rational(r1) => {
                for a in &mut self.args {
                    if let A::Rational(r2) = a.get() {
                        *a = A::Rational(r1.clone() + r2).into();
                        return;
                    }
                }
                self.args.push(rhs.clone())
                //if let Some(A::Rational(r2)) = self.first() {
                //    self.args[0] = Atom::Rational(r.clone() + r2).into();
                //} else {
                //    self.args.push_front(rhs.clone())
                //}
            }
            _ => self.args.push(rhs.clone()),
        }
    }

    fn flat_merge(lhs: &Expr, mut rhs: Sum) -> Sum {
        match lhs.get() {
            Atom::Sum(sum) => {
                let mut sum = sum.clone();
                rhs.args.into_iter().for_each(|a| {
                    sum.add_rhs(&a);
                });
                sum
            }
            &Atom::ZERO => {
                rhs
            }
            _ => {
                let mut res = vec![lhs.clone()];
                res.extend(rhs.args.drain(..));
                rhs.args = res;
                rhs
            }
        }
        //if let Atom::Sum(sum) = lhs.get() {
        //    let mut sum = sum.clone();
        //    rhs.args.into_iter().for_each(|a| {
        //        sum.add_rhs(&a);
        //    });
        //    sum
        //} else {
        //    let mut res = vec![lhs];
        //    res.extend(rhs.args);
        //    rhs.args = res;
        //    rhs
        //}
    }

    fn merge_args(p: &[Expr], q: &[Expr]) -> Sum {
        let res = if p.is_empty() {
            //q.into_iter().cloned().collect()
            Sum { args: q.into_iter().cloned().collect() } 
        } else if q.is_empty() {
            //p.into_iter().cloned().collect()
            Sum { args: p.into_iter().cloned().collect() } 
        } else {
            let p1 = p.first().unwrap();
            let q1 = q.first().unwrap();
            let p_rest = &p[1..];
            let q_rest = &q[1..];
            let h = Sum::reduce_rec(&[p1.clone(), q1.clone()]).args;
            if h.is_empty() {
                Sum::merge_args(p_rest, q_rest)
            } else if h.len() == 1 {
                Sum::flat_merge(&h[0], Sum::merge_args(p_rest, q_rest))
            } else {
                let rhs = if p1 == &h[0] && q1 == &h[1] {
                    Sum::merge_args(p_rest, q)
                } else if q1 == &h[0] && p1 == &h[1] {
                    //assert!(q1 == &h[0], "{:?} != {:?}", q1, h[0]);
                    Sum::merge_args(p, q_rest)
                } else {
                    panic!("Illegal reduction: {q:?} + {p:?} -> h")
                };

                Sum::flat_merge(&h[0], rhs)
            }
        };
        res
    }

    fn reduce_rec(args: &[Expr]) -> Sum {
        let res = if args.len() < 2 {
            //return args.into_iter().cloned().collect();
            Sum { args: args.into_iter().cloned().collect() }
        } else if args.len() == 2 {
            let lhs = &args[0];
            let rhs = &args[1];
            if let (Atom::Sum(p1), Atom::Sum(p2)) = (lhs.get(), rhs.get()) {
                Sum::merge_args(&p1.args, &p2.args)
            } else if let Atom::Sum(p) = lhs.get() {
                Sum::merge_args(&p.args, slice::from_ref(rhs))
            } else if let Atom::Sum(p) = rhs.get() {
                Sum::merge_args(slice::from_ref(lhs), &p.args)
            } else if lhs.is_const() || rhs.is_const() {
                //return SmallVec::from([Sum::reduce_mul(lhs, rhs)]);
                //Sum::reduce_mul(&lhs, &rhs)
                let mut lhs = lhs;
                let mut rhs = rhs;
                if rhs < lhs {
                    std::mem::swap(&mut lhs, &mut rhs);
                }

                let mut res = Sum::zero();
                res.add_rhs(lhs);
                res.add_rhs(rhs);
                res
            //} else if lhs.base() == rhs.base() {
            //    let e = (lhs.exponent() + rhs.exponent()).reduce();
            //    Sum { args: vec![Expr::pow(lhs.base(), e).reduce()] }
            } else {
                let mut lhs = lhs.clone();
                let mut rhs = rhs.clone();
                if rhs < lhs {
                    std::mem::swap(&mut lhs, &mut rhs);
                }

                Sum { args: vec![lhs, rhs] }
            }
        } else {
            let lhs = args.first().unwrap();
            let rhs = Sum::reduce_rec(&args[1..]);

            if let Atom::Sum(p) = lhs.get() {
                Sum::merge_args(&p.args, &rhs.args)
            } else {
                Sum::merge_args(slice::from_ref(lhs), &rhs.args)
            }
        };
        res
    }

    fn reduce(&self) -> Expr {
        let mut sum = Sum::reduce_rec(&self.args);
        if sum.is_zero() {
            Expr::zero()
        } else if sum.is_undef() {
            Expr::undef()
        } else if sum.args.len() == 1 {
            sum.args.remove(0)
        } else {
            Atom::Sum(sum).into()
        }
    }
}
impl fmt::Debug for Sum {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.args.is_empty() {
            return write!(f, "(+)");
        } else if self.args.len() == 1 {
            return write!(f, "(+{:?})", self.args[0]);
        }
        utils::fmt_iter(
            ["(", " + ", ")"],
            self.args.iter(),
            |a, f| write!(f, "{a:?}"),
            f,
        )
    }
}

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Prod {
    pub(crate) args: Vec<Expr>,
}
impl fmt::Debug for Prod {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.args.is_empty() {
            return write!(f, "(*)");
        } else if self.args.len() == 1 {
            return write!(f, "(1*{:?})", self.args[0])
        }
        utils::fmt_iter(
            ["(", " * ", ")"],
            self.args.iter(),
            |a, f| write!(f, "{a:?}"),
            f,
        )
    }
}

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct ReducedProd {
    lhs: Expr,
    rhs: Option<Expr>,
}

impl From<Expr> for ReducedProd {
    fn from(value: Expr) -> Self {
        Self {
            lhs: value,
            rhs: None,
        }
    }
}

impl Prod {
    pub fn one() -> Self {
        Prod {
            args: Default::default(),
        }
    }

    pub fn is_one(&self) -> bool {
        self.args.is_empty()
    }

    pub fn is_undef(&self) -> bool {
        self.args.first().is_some_and(|e| e.is_undef())
    }

    fn first(&self) -> Option<&Atom> {
        self.args.first().map(|a| a.get())
    }

    fn as_binary_mul(&self) -> (Expr, Expr) {
        if self.args.is_empty() {
            (Expr::one(), Expr::one())
        } else if self.args.len() == 1 {
            (self.args[0].clone(), Expr::one())
        } else {
            let a = self.args.first().unwrap().clone();
            let rest = &self.args[1..];
            let b = if rest.len() == 1 {
                rest[0].clone()
            } else {
                Expr::from(Atom::Prod(Prod {
                    args: rest.into_iter().cloned().collect(),
                }))
            };
            (a, b)
        }
    }

    pub fn mul_rhs(&mut self, rhs: &Expr) {
        use Atom as A;

        if let Atom::Undef = rhs.get() {
            self.args.clear();
            self.args.push(rhs.clone());
            return;
        }

        match self.first() {
            Some(A::Undef | &A::ZERO) => return,
            Some(_) | None => (),
        }

        match rhs.get() {
            &A::ONE => (),
            A::Undef | &A::ZERO => {
                self.args.clear();
                self.args.push(rhs.clone())
            }
            A::Prod(prod) => {
                prod.args.iter().for_each(|a| self.mul_rhs(a));
            }
            //A::Rational(r) => {
            //    if let Some(A::Rational(r2)) = self.first() {
            //        self.args[0] = A::Rational(r.clone() * r2).into();
            //    } else {
            //        self.args.push(rhs.clone())
            //    }
            //}
            A::Rational(r1) => {
                for a in &mut self.args {
                    if let A::Rational(r2) = a.get() {
                        *a = A::Rational(r1.clone() * r2).into();
                        return;
                    }
                }
                self.args.push(rhs.clone())
            }
            A::Irrational(_) | A::Func(_) | A::Var(_) | A::Sum(_) | A::Pow(_) => {
                if let Some(arg) = self.args.iter_mut().find(|a| a.base() == rhs.base()) {
                    *arg = Expr::pow(arg.base(), arg.exponent() + rhs.exponent())
                } else {
                    self.args.push(rhs.clone())
                }
            }
        }
    }

    //fn reduce_mul(lhs: &Expr, rhs: &Expr) -> Expr {
    //    match (lhs.get(), rhs.get()) {
    //        (Atom::Undef, _) | (_, Atom::Undef) => Expr::undef(),
    //        (&Atom::ZERO, _) | (_, &Atom::ZERO) => Expr::zero(),
    //        (&Atom::ONE, _) => rhs.clone(),
    //        (_, &Atom::ONE) => lhs.clone(),
    //        (Atom::Rational(r1), Atom::Rational(r2)) => Atom::Rational(r1.clone() * r2).into(),
    //        _ => Atom::Prod(Prod {
    //            args: [lhs.clone(), rhs.clone()].into_iter().collect(),
    //        })
    //        .into(),
    //    }
    //}

    fn expand_mul(lhs: &Expr, rhs: &Expr) -> Expr {
        use Atom as A;
        match (lhs.get(), rhs.get()) {
            (A::Prod(_), A::Prod(_)) => {
                let mut res = Prod::one();
                res.mul_rhs(lhs);
                res.mul_rhs(rhs);
                A::Prod(res).into()
            }
            (A::Sum(sum), _) => {
                let mut res = Sum::zero();
                for term in &sum.args {
                    let term_prod = Self::expand_mul(term, rhs);
                    res.add_rhs(&term_prod)
                }
                A::Sum(res).into()
            }
            (_, A::Sum(sum)) => {
                let mut res = Sum::zero();
                for term in &sum.args {
                    let term_prod = Self::expand_mul(lhs, term);
                    res.add_rhs(&term_prod)
                }
                A::Sum(res).into()
            }
            _ => Expr::mul(lhs, rhs),
        }
    }

    fn distribute_first(&self) -> Expr {
        // prod = a * sum * b
        let sum_indx = if let Some(indx) = self
            .args
            .iter()
            .position(|a| matches!(a.get(), Atom::Sum(_)))
        {
            indx
        } else {
            return Atom::Prod(self.clone()).into();
        };

        let mut a = Prod::one();
        let mut b = Prod::one();
        let sum = self.args[sum_indx].clone();

        for (i, arg) in self.args.iter().enumerate() {
            if i < sum_indx {
                a.args.push(arg.clone())
            } else if i > sum_indx {
                b.args.push(arg.clone())
            }
        }

        let (lhs, rhs): (Expr, Expr) = (Atom::Prod(a).into(), Atom::Prod(b).into());
        let mut res = Sum::zero();

        for term in sum.iter_args() {
            res.add_rhs(&(lhs.clone() * term * rhs.clone()));
        }

        Atom::Sum(res).into()
    }

    fn distribute(&self) -> Expr {
        self.args
            .iter()
            .fold(Atom::Prod(Prod::one()).into(), |l, r| {
                Self::expand_mul(&l, r)
            })
    }

    fn flat_merge(lhs: &Expr, mut rhs: Prod) -> Prod {
        match lhs.get() {
            Atom::Prod(prod) => {
                let mut prod = prod.clone();
                rhs.args.into_iter().for_each(|a| {
                    prod.mul_rhs(&a);
                });
                prod
            }
            &Atom::ONE => {
                rhs
            }
            &Atom::ZERO => {
                rhs.args = vec![Expr::zero()];
                rhs
            }
            _ => {
                let mut res = vec![lhs.clone()];
                res.extend(rhs.args.drain(..));
                rhs.args = res;
                rhs
            }
        }
        //if let Atom::Prod(prod) = lhs.get() {
        //    let mut prod = prod.clone();
        //    rhs.args.into_iter().for_each(|a| {
        //        prod.mul_rhs(&a);
        //    });
        //    //vec![Atom::Prod(prod).into()]
        //    prod
        //} else {
        //    let mut res = vec![lhs];
        //    res.extend(rhs.args);
        //    rhs.args = res;
        //    rhs
        //}
    }

    fn merge_args(p: &[Expr], q: &[Expr]) -> Prod {
        if p.is_empty() {
            Prod { args: q.into_iter().cloned().collect() } 
        } else if q.is_empty() {
            Prod { args: p.into_iter().cloned().collect() } 
        } else {
            let p1 = p.first().unwrap();
            let q1 = q.first().unwrap();
            let p_rest = &p[1..];
            let q_rest = &q[1..];
            let h = Prod::reduce_rec(&[p1.clone(), q1.clone()]).args;
            if h.is_empty() {
                Prod::merge_args(p_rest, q_rest)
            } else if h.len() == 1 {
                Prod::flat_merge(&h[0], Prod::merge_args(p_rest, q_rest))
            } else {
                let rhs = if p1 == &h[0] && q1 == &h[1] {
                    Prod::merge_args(p_rest, q)
                } else if q1 == &h[0] && p1 == &h[1] {
                    //println!("h: {h:?}, p1: {p1:?}, q1: {q1:?}");
                    //assert!(q1 == &h[0], "{:?} != {:?}", q1, h[0]);
                    Prod::merge_args(p, q_rest)
                } else {
                    panic!("Illegal reduction: {q:?} * {p:?} -> h")
                };

                Prod::flat_merge(&h[0], rhs)
            }
        }
    }

    fn reduce_rec(args: &[Expr]) -> Prod {
        if args.len() < 2 {
            //return args.into_iter().cloned().collect();
            Prod { args: args.into_iter().cloned().collect() }
        } else if args.len() == 2 {
            let lhs = &args[0];
            let rhs = &args[1];
            if let (Atom::Prod(p1), Atom::Prod(p2)) = (lhs.get(), rhs.get()) {
                Prod::merge_args(&p1.args, &p2.args)

            } else if let Atom::Prod(p) = lhs.get() {
                Prod::merge_args(&p.args, slice::from_ref(rhs))

            } else if let Atom::Prod(p) = rhs.get() {
                Prod::merge_args(slice::from_ref(lhs), &p.args)

            } else if lhs.is_const() || rhs.is_const() {
                let mut res = Prod::one();
                if lhs < rhs {
                    res.mul_rhs(lhs);
                    res.mul_rhs(rhs);
                } else {
                    res.mul_rhs(rhs);
                    res.mul_rhs(lhs);
                }
                res
            } else if lhs.base() == rhs.base() {
                let e = (lhs.exponent() + rhs.exponent()).reduce();
                Prod { args: vec![Expr::pow(lhs.base(), e).reduce()] }
            } else {
                let mut lhs = lhs.clone();
                let mut rhs = rhs.clone();
                if rhs < lhs {
                    std::mem::swap(&mut lhs, &mut rhs);
                }

                Prod { args: vec![lhs, rhs] }
            }
        } else {
            let lhs = args.first().unwrap();
            let rhs = Prod::reduce_rec(&args[1..]);

            if let Atom::Prod(p) = lhs.get() {
                Prod::merge_args(&p.args, &rhs.args)
            } else {
                Prod::merge_args(slice::from_ref(lhs), &rhs.args)
            }
        }
    }

    fn reduce(&self) -> Expr {
        let mut prod = Prod::reduce_rec(&self.args);
        if prod.is_one() {
            Expr::one()
        } else if prod.is_undef() {
            Expr::undef()
        } else if prod.args.len() == 1 {
            prod.args.remove(0)
        } else {
            Atom::Prod(prod).into()
        }
        
        //if args.is_empty() {
        //    return Expr::one();
        //} else if args.len() == 1 {
        //    return args.pop().unwrap();
        //} else {
        //    Atom::Prod(Prod {
        //        args: args.into_iter().collect(),
        //    })
        //    .into()
        //}
    }
}

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Pow {
    /// [base, exponent]
    pub(crate) args: [Expr; 2],
}

impl Pow {
    pub fn base(&self) -> &Expr {
        &self.args[0]
    }

    pub fn exponent(&self) -> &Expr {
        &self.args[1]
    }

    pub fn expand_pow_rec(&self, recurse: bool) -> Expr {
        use Atom as A;
        let expand_pow = |lhs: &Expr, rhs: &Expr| -> Expr {
            if recurse {
                Expr::pow(lhs, rhs).expand()
            } else {
                Expr::pow(lhs, rhs)
            }
        };
        let expand_mul = |lhs: &Expr, rhs: &Expr| -> Expr {
            if recurse {
                Expr::mul(lhs, rhs).expand()
            } else {
                Expr::mul(lhs, rhs)
            }
        };

        let (e, base) = match (self.base().get(), self.exponent().get()) {
            (A::Sum(sum), A::Rational(r))
                if r.is_int() && r > &Rational::ONE && sum.args.len() > 1 =>
            {
                (r.numer().clone(), sum)
            }
            (A::Sum(sum), A::Rational(r)) if r > &Rational::ONE && sum.args.len() > 1 => {
                let (div, rem) = r.div_rem();
                return expand_mul(
                    &expand_pow(self.base(), &Expr::from(div)),
                    &expand_pow(self.base(), &Expr::from(rem)),
                );
            }
            (A::Prod(Prod { args }), _) => {
                return args
                    .iter()
                    .map(|a| {
                        if recurse {
                            Expr::pow(a, self.exponent()).expand()
                        } else {
                            Expr::pow(a, self.exponent())
                        }
                    })
                    .fold(Expr::one(), |prod, rhs| prod * rhs)
            }
            _ => {
                return A::Pow(self.clone()).into();
            }
        };

        // (a + b)^exp
        let exp = Expr::from(e.clone());
        let (a, b) = base.as_binary_sum();
        //let mut args = base.args.clone();
        //let a = args.pop_front().unwrap();
        //let b = A::Sum(Sum { args }).into();

        let mut res = Sum::zero();
        for k in Int::range_inclusive(Int::ZERO, e.clone()) {
            let rhs = if k == Int::ZERO {
                // 1 * a^exp
                expand_pow(&a, &exp)
            } else if &k == &e {
                // 1 * b^k
                expand_pow(&b, &Expr::from(k.clone()))
            } else {
                // a^k + b^(exp-k)
                let c = Int::binomial_coeff(&e, &k);
                let k_e = Expr::from(k.clone());

                expand_mul(
                    &Expr::from(c),
                    &expand_mul(
                        &expand_pow(&a, &k_e),
                        &expand_pow(&b, &Expr::from(e.clone() - &k)),
                    ),
                )
            };
            res.add_rhs(&rhs);
        }

        Atom::Sum(res).into()
    }

    pub fn expand_pow(&self) -> Expr {
        self.expand_pow_rec(true)
    }

    fn reduce(&self) -> Expr {
        use Atom as A;
        match (self.base().get(), self.exponent().get()) {
            (A::Undef, _) | (_, A::Undef) | (&A::ZERO, &A::ZERO) => Expr::undef(),
            (&A::ZERO, A::Rational(r)) if r.is_neg() => Expr::undef(),
            (&A::ZERO, A::Rational(_)) => Expr::zero(),
            (&A::ONE, _) => Expr::one(),
            (_, &A::ONE) => self.base().clone(),
            (A::Rational(b), A::Rational(e)) => {
                let (res, rem) = b.clone().pow(e.clone());
                if rem.is_zero() {
                    Expr::from(res)
                } else {
                    //Expr::from(res) * Expr::from(b.clone()).pow(Expr::from(rem))
                    A::Pow(self.clone()).into()
                }
            }
            //TODO: rule?
            (A::Pow(pow), _) => {
                let mut pow = pow.clone();
                pow.args[1] *= self.exponent();
                pow.reduce()
            }
            _ => A::Pow(self.clone()).into(),
        }
    }
}

impl fmt::Debug for Pow {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}^{:?}", self.args[0], self.args[1])
    }
}

//macro_rules! bit {
//    ($x:literal) => {
//        1 << $x
//    };
//    ($x:ident) => {
//        ExprFlags::$x.bits()
//    };
//}
//
//bitflags::bitflags! {
//    pub struct ExprFlags: u32 {
//        const Zero          = bit!(1);
//        const NonZero       = bit!(2);
//        const Invertible    = bit!(3);
//
//        const Scalar        = bit!(3);
//        const Complex       = bit!(4);
//        const Matrix        = bit!(5);
//    }
//}
//
//impl ExprFlags {
//    pub const fn is(&self, flag: ExprFlags) -> bool {
//        self.contains(flag)
//    }
//}

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Var(pub(crate) PTR<str>);
impl Var {
    fn len(&self) -> usize {
        self.0.len()
    }

    fn as_ref(&self) -> &str {
        self.0.as_ref()
    }
}
impl fmt::Debug for Var {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}
impl fmt::Display for Var {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Real {
    Rational(Rational),
    Irrational(Irrational),
}

impl fmt::Display for Real {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Real::Rational(r) => write!(f, "{r}"),
            Real::Irrational(i) => write!(f, "{i}"),
        }
    }
}

impl From<Rational> for Real {
    fn from(value: Rational) -> Self {
        Real::Rational(value)
    }
}
impl From<Irrational> for Real {
    fn from(value: Irrational) -> Self {
        Real::Irrational(value)
    }
}
impl From<Real> for Expr {
    fn from(value: Real) -> Self {
        match value {
            Real::Rational(r) => r.into(),
            Real::Irrational(i) => i.into(),
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Irrational {
    E,
    PI,
}

impl From<Irrational> for Expr {
    fn from(value: Irrational) -> Self {
        Expr::from(Atom::Irrational(value))
    }
}

impl fmt::Debug for Irrational {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self}")
    }
}
impl fmt::Display for Irrational {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let sym = match self {
            Irrational::E => "e",
            Irrational::PI => "pi",
        };
        write!(f, "{}", sym)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Func {
    Sin(Expr),
    Cos(Expr),
    Tan(Expr),
    Sec(Expr),
    ArcSin(Expr),
    ArcCos(Expr),
    ArcTan(Expr),
    Exp(Expr),
    Log(Real, Expr),
}

macro_rules! func_atom {
    ($name: ident) => {
        pub fn $name(e: impl Borrow<Expr>) -> Expr {
            paste! {
                Expr::from(Atom::Func(Func::[<$name:camel>](e.borrow().clone())))
            }
        }
    };
}

impl Func {
    pub fn name(&self) -> String {
        match self {
            Func::Sin(_) => "sin".into(),
            Func::Cos(_) => "cos".into(),
            Func::Tan(_) => "tan".into(),
            Func::Sec(_) => "sec".into(),
            Func::ArcSin(_) => "arcsin".into(),
            Func::ArcCos(_) => "arccos".into(),
            Func::ArcTan(_) => "arctan".into(),
            Func::Exp(_) => "exp".into(),
            Func::Log(Real::Irrational(Irrational::E), _) => "ln".into(),
            Func::Log(Real::Rational(r), _) if r == &Rational::from(10) => "log".into(),
            Func::Log(base, _) => format!("{base}"),
        }
    }
    pub fn derivative(&self, x: impl Borrow<Expr>) -> Expr {
        use Expr as E;
        use Func as F;
        let d = |e: &Expr| -> Expr { e.derivative(x) };
        match self {
            F::Sin(f) => d(f) * E::cos(f),
            F::Cos(f) => d(f) * E::from(-1) * E::sin(f),
            F::Tan(f) => d(f) * E::pow(E::sec(f), &2.into()),
            F::Sec(f) => d(f) * E::tan(f) * E::sec(f),
            F::ArcSin(f) => {
                d(f) * E::pow(E::from(1) - E::pow(f, E::from(2)), E::from((-1i64, 2i64)))
            }
            F::ArcCos(f) => {
                d(f) * E::from(-1)
                    * E::pow(E::from(1) - E::pow(f, E::from(2)), E::from((-1i64, 2i64)))
            }
            F::ArcTan(f) => d(f) * E::pow(E::from(1) + E::pow(f, E::from(2)), E::from(-1)),
            F::Log(base, f) => d(f) * Expr::pow(f * E::ln(Expr::from(base.clone())), E::from(-1)),
            F::Exp(f) => Expr::exp(f) * d(f),
        }
    }

    pub fn args(&self) -> &[Expr] {
        use Func as F;
        match self {
            F::Sin(x)
            | F::Cos(x)
            | F::Tan(x)
            | F::Sec(x)
            | F::ArcSin(x)
            | F::ArcCos(x)
            | F::ArcTan(x)
            | F::Exp(x)
            | F::Log(_, x) => slice::from_ref(x),
        }
    }

    pub fn args_mut(&mut self) -> &mut [Expr] {
        use Func as F;
        match self {
            F::Sin(x)
            | F::Cos(x)
            | F::Tan(x)
            | F::Sec(x)
            | F::ArcSin(x)
            | F::ArcCos(x)
            | F::ArcTan(x)
            | F::Exp(x)
            | F::Log(_, x) => slice::from_mut(x),
        }
    }

    pub fn iter_args(&self) -> impl Iterator<Item = &Expr> {
        self.args().iter()
    }
    pub fn iter_args_mut(&mut self) -> impl Iterator<Item = &mut Expr> {
        self.args_mut().iter_mut()
    }

    pub fn reduce(&self) -> Expr {
        use Func as F;
        match self {
            F::Log(base, x) if x.check_real(base) => Expr::one(),
            _ => Atom::Func(self.clone()).into(),
        }
    }
}

//#[derive(Debug, Clone, PartialEq, Eq, Hash)]
//pub struct Func {
//    kind: FuncKind,
//    args: Vec<Expr>,
//}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct Derivative {
    arg: Expr,
    var: Expr,
    degree: u64,
}

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Atom {
    Undef,
    Rational(Rational),
    Irrational(Irrational),
    Var(Var),
    Sum(Sum),
    Prod(Prod),
    Pow(Pow),
    Func(Func),
}

impl From<Atom> for Expr {
    fn from(value: Atom) -> Self {
        Expr(value.into())
    }
}


impl Atom {
    pub const MINUS_ONE: Atom = Atom::Rational(Rational::MINUS_ONE);
    pub const ZERO: Atom = Atom::Rational(Rational::ZERO);
    pub const ONE: Atom = Atom::Rational(Rational::ONE);
}

impl fmt::Debug for Atom {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Atom::Undef => write!(f, "undef"),
            Atom::Rational(r) => write!(f, "{r}"),
            Atom::Irrational(r) => write!(f, "{r}"),
            Atom::Var(v) => write!(f, "{v}"),
            Atom::Sum(sum) => write!(f, "{:?}", sum),
            Atom::Prod(prod) => write!(f, "{:?}", prod),
            Atom::Pow(pow) => write!(f, "{:?}", pow),
            Atom::Func(func) => write!(f, "{func:?}"),
        }
    }
}

pub(crate) type PTR<T> = Arc<T>;

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Expr(PTR<Atom>);

//static E_ONE: std::sync::LazyLock<Expr> = std::sync::LazyLock::new(|| {
//    Expr::from(Atom::ONE)
//});
//static E_ZERO: std::sync::LazyLock<Expr> = std::sync::LazyLock::new(|| {
//    Expr::from(Atom::ZERO)
//});


// utility
impl Expr {
    pub fn undef() -> Expr {
        Atom::Undef.into()
    }

    pub fn zero() -> Expr {
        Atom::ZERO.into()
    }

    pub fn one() -> Expr {
        Atom::ONE.into()
    }

    pub fn min_one() -> Expr {
        Atom::MINUS_ONE.into()
    }

    pub fn var(str: &str) -> Expr {
        Expr::from(str)
    }

    pub fn rational<T: Into<Rational>>(r: T) -> Expr {
        Atom::Rational(r.into()).into()
    }

    pub fn add(lhs: impl Borrow<Expr>, rhs: impl Borrow<Expr>) -> Expr {
        use Atom as A;
        let (lhs, rhs): (&Expr, &Expr) = (lhs.borrow(), rhs.borrow());
        match (lhs.get(), rhs.get()) {
            (A::Undef, _) => A::Undef.into(),
            (_, A::Undef) => A::Undef.into(),
            (&A::ZERO, _) => rhs.clone(),
            (_, &A::ZERO) => lhs.clone(),
            (A::Rational(r1), A::Rational(r2)) => A::Rational(r1.clone() + r2).into(),
            (_, _) => {
                let mut sum = Sum::zero();
                sum.add_rhs(lhs);
                sum.add_rhs(rhs);
                A::Sum(sum).into()
            }
        }
    }
    pub fn sub(lhs: impl Borrow<Expr>, rhs: impl Borrow<Expr>) -> Expr {
        let (lhs, rhs) = (lhs.borrow(), rhs.borrow());
        let min_one = Expr::from(-1);
        let min_rhs = Expr::mul(min_one, rhs);
        Expr::add(lhs, min_rhs)
    }
    pub fn mul(lhs: impl Borrow<Expr>, rhs: impl Borrow<Expr>) -> Expr {
        use Atom as A;
        let (lhs, rhs): (&Expr, &Expr) = (lhs.borrow(), rhs.borrow());
        match (lhs.get(), rhs.get()) {
            (A::Undef, _) | (_, A::Undef) => A::Undef.into(),
            (&A::ZERO, _) | (_, &A::ZERO) => Expr::zero(),
            (&A::ONE, _) => rhs.clone(),
            (_, &A::ONE) => lhs.clone(),
            (A::Rational(r1), A::Rational(r2)) => A::Rational(r1.clone() * r2).into(),
            (_, _) => {
                if lhs.base() == rhs.base() {
                    return Expr::pow(lhs.base(), lhs.exponent() + rhs.exponent());
                } else {
                    let mut prod = Prod::one();
                    prod.mul_rhs(lhs);
                    prod.mul_rhs(rhs);
                    A::Prod(prod).into()
                }
                //Expr::prod([lhs, rhs]),
            }
        }
        //Expr::prod([lhs.borrow(), rhs.borrow()])
    }
    pub fn div(lhs: impl Borrow<Expr>, rhs: impl Borrow<Expr>) -> Expr {
        let min_one = Expr::from(-1);
        let inv_rhs = Expr::pow(rhs, &min_one);
        Expr::mul(lhs, &inv_rhs)
    }

    func_atom!(cos);
    func_atom!(sin);
    func_atom!(tan);
    func_atom!(sec);
    func_atom!(arc_sin);
    func_atom!(arc_cos);
    func_atom!(arc_tan);
    func_atom!(exp);

    pub fn log(base: impl Into<Real>, e: impl Borrow<Expr>) -> Expr {
        let base = base.into();
        Expr::from(Atom::Func(Func::Log(base, e.borrow().clone())))
    }
    pub fn ln(e: impl Borrow<Expr>) -> Expr {
        Expr::log(Irrational::E, e)
    }
    pub fn log10(e: impl Borrow<Expr>) -> Expr {
        Expr::log(Rational::from(10), e)
    }

    pub fn mul_raw(lhs: impl Borrow<Expr>, rhs: impl Borrow<Expr>) -> Expr {
        Atom::Prod(Prod { args: vec![lhs.borrow().clone(), rhs.borrow().clone()] }).into()
    }
    pub fn div_raw(lhs: impl Borrow<Expr>, rhs: impl Borrow<Expr>) -> Expr {
        let min_one = Expr::min_one();
        let one_div_rhs = Expr::pow_raw(rhs, min_one);
        Expr::mul_raw(lhs, one_div_rhs)
    }
    pub fn pow_raw(base: impl Borrow<Expr>, exponent: impl Borrow<Expr>) -> Expr {
        Atom::Pow(Pow {
            args: [base.borrow().clone(), exponent.borrow().clone()],
        })
        .into()
    }

    pub fn pow(base: impl Borrow<Expr>, exponent: impl Borrow<Expr>) -> Expr {
        use Atom as A;
        let (base, exponent) = (base.borrow(), exponent.borrow());
        match (base.get(), exponent.get()) {
            (A::Undef, _) | (_, A::Undef) | (&A::ZERO, &A::ZERO) => Expr::undef(),
            (&A::ZERO, A::Rational(r)) if r.is_neg() => Expr::undef(),
            (&A::ONE, _) => Expr::one(),
            (_, &A::ONE) => base.clone(),
            (_, &A::ZERO) => Expr::one(),
            (A::Rational(b), A::Rational(e)) if b.is_int() && e.is_int() => {
                let (pow, rem) = b.clone().pow(e.clone());
                assert!(rem.is_zero());
                Expr::from(pow)
            }
            (A::Pow(pow), A::Rational(e)) if e.is_int() => {
                Expr::pow(pow.base(), pow.exponent() * exponent)
            }
            //(A::Pow(pow), A::Rational(e2)) if e2.is_int() => {
            //    println!("pow: {base:?}, {exponent:?}");
            //    match pow.exponent().get() {
            //        A::Rational(e1) => {
            //            Expr::pow(base, Expr::from(e1.clone() * e2))
            //        }
            //        _ => Expr::raw_pow(base, exponent),
            //    }
            //}
            _ => Expr::pow_raw(base, exponent),
        }
    }

    pub fn sqrt(v: impl Borrow<Expr>) -> Expr {
        Expr::pow(v, Expr::from((1u64, 2u64)))
    }

    pub fn is_undef(&self) -> bool {
        matches!(self.get(), Atom::Undef)
    }

    fn is_primitive(&self) -> bool {
        use Atom as A;
        match self.get() {
            A::Irrational(_) | A::Undef | A::Rational(_) => true,
            A::Func(_) | A::Var(_) | A::Sum(_) | A::Prod(_) | A::Pow(_) => false,
        }
    }

    pub fn check_real(&self, r: &Real) -> bool {
        match (self.get(), r) {
            (Atom::Rational(r1), Real::Rational(r2)) => r1 == r2,
            (Atom::Irrational(i1), Real::Irrational(i2)) => i1 == i2,
            _ => false,
        }
    }
}

// utility
impl Expr {
    pub fn free_of(&self, expr: &Expr) -> bool {
        self.iter_compl_sub_exprs().all(|e| e != expr)
    }

    pub fn free_of_set<'a, I: IntoIterator<Item = &'a Self>>(&'a self, exprs: I) -> bool {
        exprs.into_iter().all(|e| self.free_of(e))
    }
    pub fn substitude(&self, from: &Expr, to: &Expr) -> Self {
        self.concurr_substitude([(from, to)])
    }

    pub fn seq_substitude<'a, T>(&self, subs: T) -> Self
    where
        T: IntoIterator<Item = (&'a Expr, &'a Expr)>,
    {
        let mut res = self.clone();
        subs.into_iter().for_each(|(from, to)| {
            res.for_each_compl_sub_expr(|sub_expr| {
                if sub_expr == from {
                    *sub_expr = to.clone();
                }
            });
        });
        res
    }

    pub fn concurr_substitude<'a, T>(&self, subs: T) -> Self
    where
        T: IntoIterator<Item = (&'a Expr, &'a Expr)> + Copy,
    {
        let mut res = self.clone();
        res.try_for_each_compl_sub_expr(|sub_expr| {
            for (from, to) in subs {
                if sub_expr == from {
                    *sub_expr = (*to).clone();
                    return ops::ControlFlow::Break(());
                }
            }
            ops::ControlFlow::Continue(())
        });
        res
    }

    pub fn variables(&self) -> HashSet<Expr> {
        let mut vars = Default::default();
        self.variables_impl(&mut vars);
        vars
    }

    pub fn numerator(&self) -> Expr {
        use Atom as A;
        match self.get() {
            A::Undef => self.clone(),
            A::Rational(r) => r.numer().into(),
            A::Pow(pow) => match pow.exponent().get() {
                &A::MINUS_ONE => Expr::one(),
                _ => self.clone(),
            },
            A::Prod(Prod { args }) => args
                .iter()
                .map(|a| a.numerator())
                .fold(Expr::one(), |prod, a| prod * a),
            _ => self.clone(),
        }
    }

    pub fn denominator(&self) -> Expr {
        use Atom as A;
        match self.get() {
            A::Undef => self.clone(),
            A::Rational(r) => r.denom().into(),
            A::Pow(pow) => match pow.exponent().get() {
                &A::MINUS_ONE => pow.base().clone(),
                _ => Expr::one(),
            },
            A::Prod(Prod { args }) => args
                .iter()
                .map(|a| a.denominator())
                .fold(Expr::one(), |prod, a| prod * a),
            _ => Expr::one(),
        }
    }

    pub fn base(&self) -> Expr {
        match self.get_flatten() {
            Atom::Pow(p) => p.base().clone(),
            _ => self.clone(),
        }
    }

    pub fn exponent(&self) -> Expr {
        match self.get_flatten() {
            Atom::Pow(p) => p.exponent().clone(),
            _ => Expr::one(),
        }
    }

    fn is_const(&self) -> bool {
        match self.get() {
            Atom::Undef | Atom::Rational(_) | Atom::Irrational(_) => true,
            _ => false,
        }
    }

    pub fn r#const(&self) -> Option<Expr> {
        match self.get() {
            Atom::Var(_) | Atom::Sum(_) | Atom::Pow(_) | Atom::Func(_) => Some(Expr::one()),
            Atom::Undef | Atom::Rational(_) | Atom::Irrational(_) => None,
            Atom::Prod(Prod { args }) => {
                if let Some(a) = args.first() {
                    if a.is_const() {
                        return Some(a.clone());
                    }
                }
                Some(Expr::one())
            }
        }
    }

    pub fn term(&self) -> Option<Expr> {
        match self.get() {
            Atom::Prod(prod) => {
                if let Some(a) = prod.args.first() {
                    if a.is_const() {
                        let (_, c) = prod.as_binary_mul();
                        return Some(c);
                        //let mut args = args.clone();
                        //args.pop_front();
                        //if args.is_empty() {
                        //    return Some(Expr::one());
                        //} else if args.len() == 1 {
                        //    return Some(args.pop_back().unwrap());
                        //} else {
                        //    return Some(Atom::Prod(Prod { args }).into());
                        //}
                    }
                }
                Some(self.clone())
            }
            Atom::Var(_) | Atom::Sum(_) | Atom::Pow(_) | Atom::Func(_) => Some(self.clone()),
            Atom::Undef | Atom::Rational(_) | Atom::Irrational(_) => None,
        }
    }

    pub fn get(&self) -> &Atom {
        self.0.as_ref()
    }

    pub fn get_flatten(&self) -> &Atom {
        match self.0.as_ref() {
            Atom::Sum(Sum { args }) | Atom::Prod(Prod { args }) if args.len() == 1 => {
                args.first().unwrap().get()
            }
            a => a,
        }
    }

    pub fn make_mut(&mut self) -> &mut Atom {
        PTR::make_mut(&mut self.0)
    }

    pub fn args(&self) -> &[Self] {
        use Atom as A;
        match self.get() {
            A::Undef
            | A::Rational(_)
            | A::Irrational(_)
            | A::Var(_) => &[],
            A::Sum(Sum { args })
            | A::Prod(Prod { args }) => args.as_slice(),
            A::Pow(Pow { args }) => args,
            A::Func(func) => func.args(),
        }
    }

    pub fn args_mut(&mut self) -> &mut [Self] {
        use Atom as A;
        match self.make_mut() {
            A::Undef
            | A::Rational(_)
            | A::Irrational(_)
            | A::Var(_) => &mut [],
            A::Sum(Sum { args })
            | A::Prod(Prod { args }) => args.as_mut_slice(),
            A::Pow(Pow { args }) => args,
            A::Func(func) => func.args_mut(),
        }
    }

    fn iter_args(&self) -> impl Iterator<Item = &Self> {
        self.args().iter()
    }

    fn iter_args_mut(&mut self) -> impl Iterator<Item = &mut Self> {
        self.args_mut().iter_mut()
    }

    fn map_args(mut self, map_fn: impl Fn(&mut Expr)) -> Self {
        self.iter_args_mut().for_each(map_fn);
        self
    }

    fn try_for_each_compl_sub_expr<F>(&mut self, func: F)
    where
        F: Fn(&mut Expr) -> ops::ControlFlow<()> + Copy,
    {
        if func(self).is_break() {
            return;
        }

        self.iter_args_mut()
            .for_each(|a| a.try_for_each_compl_sub_expr(func))
    }

    fn for_each_compl_sub_expr<F>(&mut self, func: F)
    where
        F: Fn(&mut Expr) + Copy,
    {
        self.try_for_each_compl_sub_expr(|expr| {
            func(expr);
            ops::ControlFlow::Continue(())
        });
    }

    fn iter_compl_sub_exprs(&self) -> ExprIterator<'_> {
        let atoms = vec![self];
        ExprIterator { atoms }
    }

    fn variables_impl(&self, vars: &mut HashSet<Expr>) {
        use Atom as A;
        match self.get() {
            A::Irrational(_) | A::Rational(_) | A::Undef => (),
            A::Var(_) => {
                vars.insert(self.clone());
            }
            A::Sum(Sum { args }) => args.iter().for_each(|a| a.variables_impl(vars)),
            A::Prod(Prod { args }) => {
                args.iter().for_each(|a| {
                    if let A::Sum(_) = a.get() {
                        vars.insert(a.clone());
                    } else {
                        a.variables_impl(vars)
                    }
                });
            }
            A::Pow(pow) => {
                if let A::Rational(r) = pow.exponent().get() {
                    if r >= &Rational::ONE {
                        vars.insert(pow.base().clone());
                        return;
                    }
                }
                vars.insert(self.clone());
            }
            A::Func(_) => todo!(),
        }
    }
}

impl Expr {
    /*
    pub fn pow_expand(base: impl Borrow<Expr>, exponent: impl Borrow<Expr>) -> Expr {
        Expr::pow(base, exponent).expand()
    }

    pub fn mul_expand(lhs: impl Borrow<Expr>, rhs: impl Borrow<Expr>) -> Expr {
        use Atom as A;
        let (lhs, rhs) = (lhs.borrow(), rhs.borrow());
        match (lhs.get(), rhs.get()) {
            (A::Prod(_), A::Prod(_)) => {
                let mut res = Prod::one();
                res.mul_rhs(lhs);
                res.mul_rhs(rhs);
                A::Prod(res).into()
            }
            (A::Sum(sum), _) => {
                let mut res = Sum::zero();
                for term in &sum.args {
                    let term_prod = Self::mul_expand(term, rhs);
                    res.add_rhs(&term_prod)
                }
                A::Sum(res).into()
            }
            (_, A::Sum(sum)) => {
                let mut res = Sum::zero();
                for term in &sum.args {
                    let term_prod = Self::mul_expand(lhs, term);
                    res.add_rhs(&term_prod)
                }
                A::Sum(res).into()
            }
            _ => Expr::mul(lhs, rhs),
        }
    }
    */

    pub fn derivative<T: Borrow<Self>>(&self, x: T) -> Self {
        use Atom as A;
        let x = x.borrow();

        if self == x && !self.is_primitive() {
            return Expr::one();
        }

        match self.get() {
            A::Undef => self.clone(),
            A::Irrational(_) | A::Rational(_) => Expr::zero(),
            A::Sum(Sum { args }) => {
                let mut res = Sum::zero();
                args.iter()
                    .map(|a| a.derivative(x))
                    .for_each(|a| res.add_rhs(&a));
                Atom::Sum(res).into()
            }
            A::Prod(prod) => {
                if prod.args.is_empty() {
                    return Expr::zero();
                } else if prod.args.len() == 1 {
                    return prod.args.first().unwrap().derivative(x);
                }
                // prod = term * args
                //let mut args = args.clone();
                //let term = args.pop_front().unwrap();
                //let rest = Atom::Prod(Prod { args }).into();
                let (term, rest) = prod.as_binary_mul();

                // d(a * b)/dx = da/dx * b + a * db/dx
                term.derivative(x) * &rest + term * rest.derivative(x)
            }
            A::Pow(pow) => {
                let v = pow.base();
                let w = pow.exponent();
                // d(v^w)/dx = w * v^(w - 1) * dv/dx + dw/dx * v^w * ln(v)
                w * Expr::pow(v, w - Expr::one()) * v.derivative(x)
                    + w.derivative(x) * Expr::pow(v, w) * Expr::ln(v)
            }
            A::Var(_) => {
                if self.free_of(x) {
                    Expr::zero()
                } else {
                    todo!()
                }
            }
            A::Func(f) => f.derivative(x),
        }
    }

    pub fn expand(&self) -> Self {
        use Atom as A;
        let mut expanded = self.clone();
        expanded.iter_args_mut().for_each(|a| *a = a.expand());
        match expanded.get() {
            A::Var(_) | A::Undef | A::Rational(_) => expanded.clone(),

            A::Sum(Sum { args }) if args.len() == 1 => args.first().unwrap().expand(),
            A::Prod(Prod { args }) if args.len() == 1 => args.first().unwrap().expand(),
            A::Prod(prod) => prod.distribute(),
            A::Pow(pow) => match pow.exponent().get() {
                A::Rational(r) if /*r.is_int() &&*/ r > &Rational::ONE || r == &Rational::MINUS_ONE => pow.expand_pow(),
                A::Rational(r) if r == &Rational::ONE => return pow.base().clone(),
                _ => expanded.clone(),
            },
            _ => expanded.clone(),
        }
    }

    pub fn expand_main_op(&self) -> Self {
        use Atom as A;
        match self.get() {
            A::Prod(prod) => prod.distribute(),
            A::Pow(pow) => pow.expand_pow_rec(false),
            A::Irrational(_) | A::Undef | A::Rational(_) | A::Var(_) | A::Sum(_) => self.clone(),
            A::Func(_) => todo!(),
        }
    }

    pub fn distribute(&self) -> Self {
        use Atom as A;
        if let A::Prod(prod) = self.get() {
            prod.distribute_first()
        } else {
            self.clone()
        }
    }

    pub fn reduce(&self) -> Self {
        use Atom as A;
        let mut res = self.clone();
        res.iter_args_mut().for_each(|a| *a = a.reduce());

        match res.get() {
            A::Irrational(_) | A::Undef | A::Rational(_) | A::Var(_) => res,
            A::Sum(sum) => sum.reduce(),
            A::Prod(prod) => prod.reduce(),
            A::Pow(pow) => pow.reduce(),
            A::Func(func) => func.reduce(),
        }
    }

    #[inline(always)]
    pub fn as_monomial<'a>(&'a self, vars: &'a VarSet) -> MonomialView<'a> {
        MonomialView::new(self, vars)
    }
    #[inline(always)]
    pub fn as_polynomial<'a>(&'a self, vars: &'a VarSet) -> PolynomialView<'a> {
        PolynomialView::new(self, vars)
    }

    fn rationalize_add(lhs: &Self, rhs: &Self) -> Expr {
        let ln = lhs.numerator();
        let ld = lhs.denominator();
        let rn = rhs.numerator();
        let rd = rhs.denominator();
        if ld.get() == &Atom::ONE && rd.get() == &Atom::ONE {
            lhs + rhs
        } else {
            Self::rationalize_add(&(ln * &rd), &(rn * &ld)) / (ld * rd)
        }
    }

    pub fn rationalize(&self) -> Expr {
        use Atom as A;
        match self.get() {
            A::Prod(_) => {
                let mut r = self.clone();
                r.iter_args_mut().for_each(|a| *a = a.rationalize());
                r
            }
            A::Sum(Sum { args }) => args
                .iter()
                .map(|a| a.rationalize())
                .fold(Expr::zero(), |sum, r| Self::rationalize_add(&sum, &r)),
            A::Pow(pow) => Expr::pow(pow.base().rationalize(), pow.exponent()),
            _ => self.clone(),
        }
    }

    /// divide lhs and rhs by their common factor and
    /// return them in the form (fac, (lhs/fac, rhs/fac)
    pub fn factorize_common_terms(lhs: &Expr, rhs: &Self) -> (Expr, (Expr, Expr)) {
        use Atom as A;
        if lhs == rhs {
            return (lhs.clone(), (Expr::one(), Expr::one()));
        }
        match (lhs.get(), rhs.get()) {
            (A::Rational(r1), A::Rational(r2)) if r1.is_int() && r2.is_int() => {
                let (i1, i2) = (r1.to_int().unwrap(), r2.to_int().unwrap());
                let gcd = i1.gcd(&i2);
                let rgcd = Rational::from(gcd);
                let l = (r1.clone() / &rgcd).unwrap();
                let r = (r2.clone() / &rgcd).unwrap();
                (rgcd.into(), (l.into(), r.into()))
            }
            (A::Prod(prod), _) if !prod.args.is_empty() => {
                if prod.args.len() == 1 {
                    let lhs = prod.args.first().unwrap();
                    return Self::factorize_common_terms(lhs, rhs);
                }
                /*
                (a*x) * (b*y), (u*x*y)
                => common(a*x, u*x*y) -> (x, (a, u*y))
                => common(b*y, u*y) -> (y, (b, u))
                => return (x*y, (a*b, u))

                */
                //let mut args = args.clone();
                let uxy = rhs;
                let (ax, by) = prod.as_binary_mul();

                let (x, (a, uy)) = Self::factorize_common_terms(&ax, uxy);
                let (y, (b, u)) = Self::factorize_common_terms(&by, &uy);
                (x * y, (a * b, u))
            }
            (A::Sum(sum), _) if !sum.args.is_empty() => {
                if sum.args.len() == 1 {
                    let lhs = sum.args.first().unwrap();
                    return Self::factorize_common_terms(lhs, rhs);
                }
                /*
                abxy + cdxy, u*x*y*a*c
                => common(abxy, uacxy) -> (axy, (b, uc))
                => common(cdxy, uacxy) -> (cxy, (d, ua))
                => common(axy, cxy)    -> (xy, (a, c))
                => common(ua, uc)      -> (u, (a, c))
                => return (xy, (ab + cd, uac))
                */
                //let mut args = args.clone();
                let uacxy = rhs;
                let (abxy, cdxy) = sum.as_binary_sum();

                let (axy, (b, uc)) = Self::factorize_common_terms(&abxy, uacxy);
                let (cxy, (d, ua)) = Self::factorize_common_terms(&cdxy, uacxy);
                let (xy, (a, c)) = Self::factorize_common_terms(&axy, &cxy);
                let (u, (_a, _c)) = Self::factorize_common_terms(&ua, &uc);
                /*
                println!("abxy + cdxy       : {lhs:?}");
                println!("uacxy             : {uacxy:?}");
                println!("abxy              : {abxy:?}");
                println!("cdxy              : {cdxy:?}");
                println!("(axy, (b, uc))    : ({axy:?}, ({b:?}, {uc:?}))");
                println!("(cxy, (d, ua))    : ({cxy:?}, ({d:?}, {ua:?}))");
                println!("(xy, (a, c))      : ({xy:?}, ({a:?}, {c:?}))");
                println!("(u, (_a, _c))     : ({u:?}, ({_a:?}, {_c:?}))");
                println!("");
                */
                (xy, (a * b + c * d, u * _a * _c))
            }
            (_, A::Sum(_) | A::Prod(_)) => {
                let (fac, (r, l)) = Self::factorize_common_terms(rhs, lhs);
                (fac, (l, r))
            }
            (_, _) => match (lhs.exponent().get(), rhs.exponent().get()) {
                (A::Rational(r1), A::Rational(r2))
                    if r1.is_pos() && r2.is_pos() && rhs.base() == lhs.base() =>
                {
                    let e = std::cmp::min(r1, r2).clone();
                    let b = rhs.base();
                    (
                        Expr::pow(&b, Expr::from(e.clone())),
                        (
                            Expr::pow(&b, Expr::from(r1.clone() - &e)),
                            Expr::pow(&b, Expr::from(r2.clone() - e)),
                        ),
                    )
                }
                _ => (Expr::one(), (lhs.clone(), rhs.clone())),
            },
        }
    }

    pub fn common_factors(lhs: &Self, rhs: &Self) -> Expr {
        Expr::factorize_common_terms(lhs, rhs).0
    }

    pub fn factor_out(&self) -> Expr {
        use Atom as A;
        match self.get() {
            A::Prod(Prod { args }) => args
                .iter()
                .map(|a| a.factor_out())
                .fold(Expr::one(), |prod, rhs| prod * rhs),
            A::Pow(pow) => Expr::pow(pow.base().factor_out(), pow.exponent()),
            A::Sum(Sum { args }) => {
                let s = args
                    .iter()
                    .map(|a| a.factor_out())
                    .fold(Expr::zero(), |sum, rhs| sum + rhs);
                    //.reduce();
                if let A::Sum(sum) = s.get() {
                    // sum = a + b
                    let (a, b) = sum.as_binary_sum();
                    //let a = args.first().unwrap();
                    //let rest = &args[1..];
                    //let b = if rest.len() == 1 {
                    //    rest[0].clone()
                    //} else {
                    //    Expr::from(A::Sum(Sum { args: rest.into_iter().cloned().collect() }))
                    //};
                    //let mut args = args.clone();
                    //let a = args.pop_front().unwrap();
                    //let b = if args.len() == 1 {
                    //    args.pop_front().unwrap()
                    //} else {
                    //    Expr::from(A::Sum(Sum { args }))
                    //};
                    let (f, (a_div_f, b_div_f)) = Expr::factorize_common_terms(&a, &b);
                    f * (a_div_f + b_div_f)
                } else {
                    s
                }
            }
            _ => self.clone(),
        }
    }

    pub fn cancel(&self) -> Expr {
        let n = self.numerator();
        let d = self.denominator();
        n.factor_out() / d.factor_out()
    }
}

impl<T: Borrow<Expr>> ops::Add<T> for &Expr {
    type Output = Expr;
    fn add(self, rhs: T) -> Self::Output {
        Expr::add(self, rhs)
    }
}
impl<T: Borrow<Expr>> ops::Add<T> for Expr {
    type Output = Expr;
    fn add(self, rhs: T) -> Self::Output {
        Expr::add(self, rhs)
    }
}
impl<T: Borrow<Expr>> ops::AddAssign<T> for Expr {
    fn add_assign(&mut self, rhs: T) {
        *self = &*self + rhs;
    }
}
impl<T: Borrow<Expr>> ops::Sub<T> for &Expr {
    type Output = Expr;
    fn sub(self, rhs: T) -> Self::Output {
        Expr::sub(self, rhs)
    }
}
impl<T: Borrow<Expr>> ops::Sub<T> for Expr {
    type Output = Expr;
    fn sub(self, rhs: T) -> Self::Output {
        Expr::sub(self, rhs)
    }
}
impl<T: Borrow<Expr>> ops::SubAssign<T> for Expr {
    fn sub_assign(&mut self, rhs: T) {
        *self = &*self - rhs;
    }
}
impl<T: Borrow<Expr>> ops::Mul<T> for &Expr {
    type Output = Expr;
    fn mul(self, rhs: T) -> Self::Output {
        Expr::mul(self, rhs)
    }
}
impl<T: Borrow<Expr>> ops::Mul<T> for Expr {
    type Output = Expr;
    fn mul(self, rhs: T) -> Self::Output {
        Expr::mul(self, rhs)
    }
}
impl<T: Borrow<Expr>> ops::MulAssign<T> for Expr {
    fn mul_assign(&mut self, rhs: T) {
        *self = &*self * rhs;
    }
}
impl<T: Borrow<Expr>> ops::Div<T> for &Expr {
    type Output = Expr;
    fn div(self, rhs: T) -> Self::Output {
        Expr::div(self, rhs)
    }
}
impl<T: Borrow<Expr>> ops::Div<T> for Expr {
    type Output = Expr;
    fn div(self, rhs: T) -> Self::Output {
        Expr::div(self, rhs)
    }
}
impl<T: Borrow<Expr>> ops::DivAssign<T> for Expr {
    fn div_assign(&mut self, rhs: T) {
        *self = &*self / rhs;
    }
}
//impl<T: Borrow<Expr>> crate::utils::Pow<T> for &Expr {
//    type Output = Expr;
//    fn pow(self, rhs: T) -> Self::Output {
//        Expr::pow(self, rhs)
//    }
//}
//impl<T: Borrow<Expr>> crate::utils::Pow<T> for Expr {
//    type Output = Expr;
//    fn pow(self, rhs: T) -> Self::Output {
//        Expr::pow(self, rhs)
//    }
//}

impl fmt::Debug for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self.0)
    }
}
impl fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        //let fmt_ast = self.fmt_ast();
        //write!(f, "{}", self.fmt_unicode())
        write!(f, "{}", fmt_ast::FmtAtom::from(self.get()))
    }
}

impl From<&str> for Expr {
    fn from(value: &str) -> Self {
        Expr::from(Atom::Var(Var(PTR::from(value))))
    }
}
impl<T: Into<Rational>> From<T> for Expr {
    fn from(value: T) -> Self {
        Expr::from(Atom::Rational(value.into()))
    }
}

#[derive(Debug)]
struct ExprIterator<'a> {
    atoms: Vec<&'a Expr>,
}

impl<'a> Iterator for ExprIterator<'a> {
    type Item = &'a Expr;

    fn next(&mut self) -> Option<Self::Item> {
        self.atoms.pop().map(|expr| {
            expr.iter_args().for_each(|arg| self.atoms.push(arg));
            expr
        })
    }
}

trait ArgsIterator<T>: iter::ExactSizeIterator<Item = T> {}
impl<I, T: iter::ExactSizeIterator<Item = I>> ArgsIterator<I> for T {}

trait ArgsRef: ops::Index<usize> {
    fn args(&self) -> Box<dyn ArgsIterator<&Expr> + '_>;
    fn args_mut(&mut self) -> Box<dyn ArgsIterator<&mut Expr> + '_>;
    fn len(&self) -> usize;
}

impl ArgsRef for [Expr; 2] {
    fn args(&self) -> Box<dyn ArgsIterator<&Expr> + '_> {
        Box::new(self.iter())
    }

    fn args_mut(&mut self) -> Box<dyn ArgsIterator<&mut Expr> + '_> {
        Box::new(self.iter_mut())
    }

    fn len(&self) -> usize {
        2
    }
}

impl ArgsRef for Vec<Expr> {
    fn args(&self) -> Box<dyn ArgsIterator<&Expr> + '_> {
        Box::new(self.iter())
    }

    fn args_mut(&mut self) -> Box<dyn ArgsIterator<&mut Expr> + '_> {
        Box::new(self.iter_mut())
    }

    fn len(&self) -> usize {
        Self::len(self)
    }
}

#[cfg(test)]
mod test {
    use assert_eq as eq;
    use calcurs_macros::expr as e;

    use super::*;

    #[test]
    fn variables() {
        eq!(
            e!(x ^ 3 + 3 * x ^ 2 * y + 3 * x * y ^ 2 + y ^ 3).variables(),
            [e!(x), e!(y)].into_iter().collect()
        );
        eq!(
            e!(3 * x * (x + 1) * y ^ 2 * z ^ n).variables(),
            [e!(x), e!(x + 1), e!(y), e!(z ^ n)].into_iter().collect()
        );
        eq!(
            e!(2 ^ (1 / 2) * x ^ 2 + 3 ^ (1 / 2) * x + 5 ^ (1 / 2)).variables(),
            [e!(x), e!(2 ^ (1 / 2)), e!(3 ^ (1/2)), e!(5 ^ (1 / 2))]
                .into_iter()
                .collect()
        );
    }

    #[test]
    fn expand() {
        eq!(
            e!(x * (2 + (1 + x) ^ 2)).expand_main_op(),
            e!(x * 2 + x * (1 + x) ^ 2)
        );
        eq!(
            e!((x + (1 + x) ^ 2) ^ 2).expand_main_op(),
            e!(x ^ 2 + 2 * x * (1 + x) ^ 2 + (1 + x) ^ 4)
        );
        eq!(
            e!((x + 2) * (x + 3) * (x + 4))
                .expand()
                .as_polynomial(&[e!(x)].into())
                .collect_terms(),
            Some(e!(x ^ 3 + 9 * x ^ 2 + 26 * x + 24))
        );
        eq!(
            e!((x + 1) ^ 2 + (y + 1) ^ 2).expand().reduce(),
            e!(2 + 2*x + 2*y + x^2 + y^2),
            //e!(2 + (2 * x) + x ^ 2 + (2 * y) + y ^ 2)
        );
        //eq!(
        //    e!(((x + 2) ^ 2 + 3) ^ 2).expand().reduce(),
        //    e!(x ^ 4 + 8 ^ 3 + 30 * x ^ 2 + 56 * x + 49)
        //);
    }

    #[test]
    fn reduce() {
        let checks = vec![
            (e!(2 * x), e!(2 * x)),
            (e!(1 + 2), e!(3)),
            (e!(a + undef), e!(undef)),
            (e!(a + (b + c)), e!(a + (b + c))),
            (e!(0 - 2 * b), e!((2 - 4) * b)),
            (e!(a + 0), e!(a)),
            (e!(0 + a), e!(a)),
            (e!(1 + 2), e!(3)),
            (e!(x + 0), e!(x)),
            (e!(0 + x), e!(x)),
            (e!(0 - x), e!((4 - 5) * x)),
            (e!(x - 0), e!(x)),
            (e!(3 - 2), e!(1)),
            (e!(x * 0), e!(0)),
            (e!(0 * x), e!(0)),
            (e!(x * 1), e!(x)),
            (e!(1 * x), e!(x)),
            (e!(0 ^ 0), e!(undef)),
            (e!(0 ^ 1), e!(0)),
            (e!(0 ^ 314), e!(0)),
            (e!(1 ^ 0), e!(1)),
            (e!(314 ^ 0), e!(1)),
            (e!(314 ^ 1), e!(314)),
            (e!(x ^ 1), e!(x)),
            (e!(1 ^ x), e!(1)),
            (e!(1 ^ 314), e!(1)),
            (e!(3 ^ 3), e!(27)),
            (e!(a - b), e!(a + ((2 - 3) * b))),
            (e!(a / b), e!(a * b ^ (2 - 3))),
            (e!((x ^ (1 / 2) ^ (1 / 2)) ^ 8), e!(x ^ 2)),
        ];
        for (calc, res) in checks {
            eq!(calc.reduce(), res);
        }
    }

    #[test]
    fn distributive() {
        eq!(
            e!(a * (b + c) * (d + e)).distribute(),
            e!(a * b * (d + e) + a * c * (d + e))
        );
        eq!(
            e!((x + y) / (x * y)).distribute(),
            e!(x / (x * y) + y / (x * y))
        );
    }

    #[test]
    fn num_denom() {
        let nd = |e: Expr| (e.numerator(), e.denominator());
        eq!(
            nd(e!((2 / 3) * (x * (x + 1)) / (x + 2) * y ^ n)),
            (e!(2 * x * (x + 1) * y ^ n), e!(3 * (x + 2)))
        );
    }

    #[test]
    fn rationalize() {
        eq!(e!((1 + 1 / x) ^ 2).rationalize(), e!(((x + 1) / x) ^ 2));
        eq!(
            e!((1 + 1 / x) ^ (1 / 2)).rationalize(),
            e!(((x + 1) / x) ^ (1 / 2))
        );
    }

    #[test]
    fn common_factors() {
        eq!(
            Expr::factorize_common_terms(&e!(6 * x * y ^ 3), &e!(2 * x ^ 2 * y * z)),
            (e!(2 * x * y), (e!(3 * y ^ 2), e!(x * z)))
        );
        eq!(
            Expr::factorize_common_terms(&e!(a * (x + y)), &e!(x + y)),
            (e!(x + y), (e!(a), e!(1)))
        );
    }

    #[test]
    fn factor_out() {
        eq!(
            e!((x^2 + x*y)^3).factor_out().expand_main_op(),
            e!(x^3 * (x+y)^3)
        );
        eq!(e!(a*(b + b*x)).factor_out(), e!(a*b*(1 + x)));
        eq!(
            e!(2 ^ (1 / 2) + 2).factor_out(),
            e!(2 ^ (1 / 2) * (1 + 2 ^ (1 / 2)))
        );
        eq!(
            e!(a*b*x + a*c*x + b*c*x).factor_out(),
            e!(x*(a*b + a*c + b*c))
        );
        eq!(e!(a/x + b/x), e!(a/x + b/x))
    }

    #[test]
    fn derivative() {
        let d = |e: Expr| {
            e.derivative(e!(x))
                .rationalize()
                .expand()
                .factor_out()
                .reduce()
                .reduce()
        };

        eq!(d(e!(x^2)), e!(2*x));
        eq!(d(e!(sin(x))), e!(cos(x)));
        eq!(d(e!(exp(x))), e!(exp(x)));
        eq!(d(e!(x * exp(x))), e!((1+x) * exp(x)));
        eq!(d(e!(ln(x))), e!(1/x));
        eq!(d(e!(1/x)), e!(-1/x^2));
        eq!(d(e!(tan(x))), e!(sec(x) ^ 2));
        eq!(d(e!(arc_tan(x))), e!(1/(1 + x^2)));
        eq!(
            d(e!(x * ln(x) * sin(x))),
            e!(x*cos(x)*ln(x) + sin(x)*ln(x) + sin(x))
            //e!(sin(x) + x*cos(x)*ln(x) + sin(x)*ln(x))
        );
        //eq!(d(e!(x ^ 2)), e!(2 * x));
        //eq!(d(exp(e!(sin(x)))), exp(e!(x)));
    }

    #[test]
    fn term_const() {
        eq!(e!(2 * y).term(), Some(e!(y)));
        eq!(e!(x * y).term(), Some(e!(x * y)));
        eq!(e!(x).r#const(), Some(e!(1)));
        eq!(e!(2 * x).r#const(), Some(e!(2)));
        eq!(e!(y * x).r#const(), Some(e!(1)));
    }
}
