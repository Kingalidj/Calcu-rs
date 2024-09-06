use crate::{
    expr::{Atom, Expr, Prod, Sum},
    rational::{Rational, Int},
    utils::HashMap,
};

type GVar = Expr;
type Coeff = Expr;
type Degree = Int;

#[derive(Debug, Default, Clone, PartialEq, Eq, Hash)]
pub struct VarSet {
    vars: Vec<GVar>,
}

impl VarSet {
    pub fn new<'a, I: IntoIterator<Item = &'a GVar>>(vars: I) -> Self {
        let vars = vars.into_iter().cloned().collect();
        Self { vars }
    }

    pub fn has(&self, v: &GVar) -> bool {
        self.vars.contains(v)
    }

    pub fn iter(&self) -> impl Iterator<Item = &GVar> {
        self.vars.iter()
    }
}

impl<I: IntoIterator<Item = GVar>> From<I> for VarSet {
    fn from(value: I) -> Self {
        let vars = value.into_iter().collect();
        Self { vars }
    }
}
impl From<GVar> for VarSet {
    fn from(value: GVar) -> Self {
        let vars = vec![value];
        Self { vars }
    }
}

/// represents the variable part of a term in a polynomial
///
/// e.g: x^3y^2 in a*b*x^3y^2
#[derive(Debug, Default, Clone, PartialEq, Eq, Hash)]
pub struct VarPow {
    var_deg: Vec<(GVar, Degree)>,
}

impl PartialOrd for VarPow {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for VarPow {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.total_deg().cmp(&other.total_deg())
    }
}

impl VarPow {
    pub fn degree_of(&self, v: &GVar) -> Option<&Degree> {
        self.find(v)
    }

    pub fn is_const(&self) -> bool {
        self.var_deg.is_empty()
    }

    pub fn total_deg(&self) -> Degree {
        self.var_deg
            .iter()
            .map(|(_, d)| d)
            .fold(Int::ZERO, |sum, d| sum + d)
    }

    fn find(&self, v: &GVar) -> Option<&Degree> {
        self.var_deg
            .iter()
            .find_map(|(var, d)| if var == v { Some(d) } else { None })
    }

    fn find_mut(&mut self, v: &GVar) -> Option<&mut Degree> {
        self.var_deg
            .iter_mut()
            .find_map(|(var, d)| if var == v { Some(d) } else { None })
    }

    fn to_expr(self) -> Expr {
        self.var_deg
            .into_iter()
            .map(|(v, d)| Expr::pow(v, Expr::from(d)))
            .fold(Expr::one(), |prod, var| prod * var)
    }

    fn mul(&mut self, v: GVar, d: Degree) {
        if let Some(degree) = self.find_mut(&v) {
            *degree += d;
        } else {
            self.var_deg.push((v, d))
        }
    }

    fn pow(&mut self, deg: &Degree) {
        self.var_deg.iter_mut().for_each(|(_, d)| *d *= deg);
    }

    fn merge(&mut self, mut other: Self) {
        if self.var_deg.len() < other.var_deg.len() {
            std::mem::swap(&mut self.var_deg, &mut other.var_deg);
        }
        other.var_deg.into_iter().for_each(|(v, d)| self.mul(v, d));
    }
}

impl<I: IntoIterator<Item = (GVar, Degree)>> From<I> for VarPow {
    fn from(value: I) -> Self {
        let mut res = VarPow::default();
        value.into_iter().for_each(|(v, d)| res.mul(v, d));
        res
    }
}

/// general polynomial expression (GPE), store elements of a sum as a coeff and var pair.
///
#[derive(Debug, Default, Clone, PartialEq, Eq, Hash)]
pub struct GPE {
    terms: Vec<(Coeff, VarPow)>,
}

impl GPE {
    pub fn find_mut(&mut self, vp: &VarPow) -> Option<&mut Coeff> {
        self.terms
            .iter_mut()
            .find_map(|(c, vp2)| if vp2 == vp { Some(c) } else { None })
    }

    pub fn sort_by_degree(&mut self) {
        self.terms.sort_unstable_by(|(_, vp1,), (_, vp2)| vp1.cmp(vp2).reverse());
    }

    pub fn add(&mut self, c: Coeff, vp: VarPow) {
        if let Some(coeff) = self.find_mut(&vp) {
            *coeff += c;
        } else {
            self.terms.push((c, vp));
        }
    }

    pub fn to_expr(mut self) -> Expr {
        self.sort_by_degree();
        self.terms.into_iter().map(|(c, vp)| c * vp.to_expr()).fold(Expr::zero(), |sum, term| sum + term)
    }
}

/// View the expression as a monomial in one or multiple generalized variables
///
pub struct MonomialView<'a> {
    monom: &'a Expr,
    vars: &'a VarSet, //&'a HashSet<&'a GVar>,
}

impl<'a> MonomialView<'a> {
    pub fn new(monom: &'a Expr, vars: &'a VarSet) -> Self {
        Self { monom, vars }
    }

    pub fn check(&self) -> bool {
        use Atom as A;
        if self.vars.has(self.monom) {
            return true;
        }

        match self.monom.get() {
            A::Undef => return false,
            A::Rational(_) | A::Var(_) | A::Sum(_) => (),
            A::Prod(Prod { args }) => {
                for a in args {
                    if !a.as_monomial(self.vars).check() {
                        return false;
                    }
                }
                return true;
            }
            A::Pow(pow) => match (pow.base(), pow.exponent().get()) {
                (base, A::Rational(r))
                    // TODO: negative exponent?
                    if self.vars.has(base) && r.is_int() && r >= &Rational::ONE =>
                {
                    return true;
                }
                _ => (),
            },
            A::Func(_) => todo!(),
        }
        self.monom.free_of_set(self.vars.iter())
    }

    pub fn degree(&self) -> Option<Degree> {
        self.coeff().map(|(_, d)| d.total_deg())
    }

    pub fn coeff(&self) -> Option<(Coeff, VarPow)> {
        use Atom as A;
        if self.monom.is_undef() {
            return None;
        }
        if self.vars.has(self.monom) {
            let v = self.monom;
            return Some((Expr::one(), [(v.clone(), Int::ONE)].into()));
        }

        match self.monom.get() {
            A::Prod(Prod { args }) => {
                let mut coeff = Expr::one();
                let mut degree = VarPow::default();
                for a in args {
                    let (c, d) = a.as_monomial(self.vars).coeff()?;
                    coeff *= c;
                    degree.merge(d)
                }
                return Some((coeff, degree));
            }
            A::Pow(pow) => {
                match pow.exponent().get() {
                    // TODO: negative exponent?
                    A::Rational(r) if r.is_int() && r >= &Rational::ONE => {
                        let (c, mut d) = pow.base().as_monomial(self.vars).coeff()?;
                        d.pow(&r.to_int().unwrap());
                        return Some((Expr::pow(c, pow.exponent()), d))
                        //if self.vars.has(pow.base()) {
                        //    let v = pow.base();
                        //    return Some((Expr::one(), [(v.clone(), r.numer().clone())].into()));
                        //}
                    }
                    _ => (),
                } 
            }
            _ => (),
        }

        if self.monom.free_of_set(self.vars.iter()) {
            Some((self.monom.clone(), Default::default()))
        } else {
            None
        }
    }

    pub fn degree_of(&self, v: &Expr) -> Option<Degree> {
        self.coeff()
            .and_then(|(_, degs)| degs.degree_of(v).cloned())
    }
}

/// View the expression as a polynomial in one or multiple generalized variables
///
pub struct PolynomialView<'a> {
    poly: &'a Expr,
    vars: &'a VarSet,
}

impl<'a> PolynomialView<'a> {
    pub fn new(poly: &'a Expr, vars: &'a VarSet) -> Self {
        Self { poly, vars }
    }

    pub fn check(&self) -> bool {
        use Atom as A;
        if let A::Sum(Sum { args }) = self.poly.get() {
            if self.vars.has(self.poly) {
                return true;
            }

            for a in args {
                if !a.as_monomial(self.vars).check() {
                    return false;
                }
            }
            true
        } else {
            self.poly.as_monomial(self.vars).check()
        }
    }

    pub fn degree(&self) -> Option<Int> {
        self.coeffs()
            .into_iter()
            .map(|(d, _)| d.total_deg())
            .reduce(|max, d| std::cmp::max(max, d))
    }

    pub fn degree_of(&self, v: &GVar) -> Option<Int> {
        self.coeffs()
            .into_iter()
            .filter_map(|(d, _)| d.degree_of(v).cloned())
            .reduce(|max, d| std::cmp::max(max, d))
    }

    pub fn coeffs_of_deg(&self, v: &GVar, deg: &Degree) -> Option<Coeff> {
        self.coeffs_of(v).remove(deg)
    }

    pub fn coeffs_of(&self, v: &GVar) -> HashMap<Degree, Coeff> {
        let coeff = self.coeffs().into_iter().filter_map(|(d, c)| {
            if let Some(d) = d.degree_of(v) {
                Some((d.clone(), c))
            } else if d.is_const() {
                Some((Int::ZERO, c))
            } else {
                None
            }
        });

        let mut res = HashMap::default();

        for (d, c) in coeff {
            res.entry(d).and_modify(|coeff| *coeff += &c).or_insert(c);
        }

        res
    }

    pub fn coeffs(&self) -> HashMap<VarPow, Coeff> {
        use Atom as A;
        let mut coeffs = HashMap::default();
        if self.poly.is_undef() {
            return coeffs;
        }

        if let A::Sum(Sum { args }) = self.poly.get() {
            if self.vars.has(self.poly) {
                let v = self.poly;
                coeffs.insert([(v.clone(), Int::ZERO)].into(), Expr::one());
                return coeffs;
            }

            for a in args {
                match a.as_monomial(self.vars).coeff() {
                    Some((c, d)) => {
                        coeffs
                            .entry(d)
                            .and_modify(|coeff| *coeff += &c)
                            .or_insert(c);
                    }
                    None => return Default::default(),
                }
            }
            //coeffs.iter_mut().for_each(|(_, c)| *c = c.reduce());
            coeffs
        } else {
            if let Some((c, d)) = self.poly.as_monomial(self.vars).coeff() {
                coeffs.insert(d, c);
            }
            coeffs
        }
    }

    pub fn leading_coeff_of(&self, v: &GVar) -> Option<Coeff> {
        self.coeffs_of(v)
            .into_iter()
            .max_by(|a, b| a.0.cmp(&b.0))
            .map(|(_, coeff)| coeff)
    }

    pub fn collect_terms(&self) -> Option<Expr> {
        use Atom as A;
        let sum = if let A::Sum(sum) = self.poly.get() {
            sum
        } else {
            // self.poly != A::Sum(_)
            if self.poly.as_monomial(self.vars).check() {
                return None;
            } else {
                return Some(self.poly.clone());
            }
        };

        if self.vars.has(self.poly) {
            return Some(self.poly.clone());
        }

        let mut res = GPE::default();
        for a in &sum.args {
            let (mc, mv) = a.as_monomial(self.vars).coeff()?;
            res.add(mc, mv);
        }
        Some(res.to_expr())

        //let mut n = 0;
        //let mut t: Vec<(Expr, VarPow)> = vec![(Expr::zero(), Default::default()); sum.args.len()];
        //for a in &sum.args {
        //    let f = a.as_monomial(self.vars).coeff()?;
        //    let mut j = 0;
        //    let mut combined = false;
        //    while !combined && j < n {
        //        if f.1 == t[j].1 {
        //            t[j] = (f.0.clone() + &t[j].0, f.1.clone());
        //            combined = true;
        //        }
        //        j += 1;
        //    }
        //    if !combined {
        //        t[n + 1] = f;
        //        n += 1;
        //    }
        //}

        //let mut v = Expr::zero();
        //for (coeff, var) in t {
        //    v += coeff * var.to_expr();
        //}
        //Some(v)
    }
}

#[cfg(test)]
mod monomial_uv {
    use calcurs_macros::expr as e;

    use super::*;

    #[test]
    fn check() {
        let x = &VarSet::from(e!(x));
        let xy = &VarSet::from([e!(x), e!(y)]);
        assert!(e!(x).as_monomial(x).check());
        assert!(e!(x ^ 1).as_monomial(x).check());
        assert!(e!(x * y).as_monomial(x).check());
        assert!(e!(2 * x).as_monomial(x).check());
        assert!(e!(x * (1 + 2)).as_monomial(x).check());
        assert!(e!(2 * x ^ 3).as_monomial(x).check());
        assert!(!e!(x + 1).as_monomial(x).check());
        assert!(!e!((x + 1) * (x + 3)).as_monomial(x).check());
        assert!(!e!(x * (x + 3)).as_monomial(x).check());
        assert!(e!(a * x ^ 2 * y ^ 2).as_monomial(xy).check());
        assert!(!e!(a * (x ^ 2 + y ^ 2)).as_monomial(xy).check());
    }

    #[test]
    fn degree() {
        let x = &VarSet::from([e!(x)]);
        assert_eq!(e!(x ^ 3).as_monomial(x).degree(), Some(3.into()));
        assert_eq!(e!(x ^ 3 * x ^ 4).as_monomial(x).degree(), Some(7.into()));
        assert_eq!(
            e!(3 * w * x ^ 2 * y ^ 3 * z ^ 4)
                .as_monomial(&[e!(x), e!(z)].into())
                .degree(),
            Some(6.into())
        )
    }

    #[test]
    fn coeff() {
        let x = &VarSet::from([e!(x)]);
        assert_eq!(
            e!(a ^ 2 * x * b * x ^ 2).as_monomial(x).coeff(),
            Some((e!(a ^ 2 * b), [(e!(x), 3.into())].into()))
        );
        assert_eq!(
            e!(a ^ 2 * x ^ 2 * x ^ 4).as_monomial(x).degree(),
            Some(6.into())
        )
    }
}

#[cfg(test)]
mod polynomial_uv {
    use calcurs_macros::expr as e;

    use super::*;

    macro_rules! p {
        ($expr:expr, $vars:expr) => {
            $expr.as_polynomial(&$vars.into())
        };
    }

    #[test]
    fn check() {
        assert!(p!(e!(3 * x ^ 2 + 4 * x + 5), e!(x)).check());
        assert!(p!(e!(a * x ^ 2 + b * x + c), e!(x)).check());
        assert!(!p!(e!(x * (x ^ 2 + 1)), e!(x)).check());
        assert!(p!(e!(x ^ 2 * (x ^ 4 + 1)), e!(x ^ 2)).check());
    }

    #[test]
    fn degree() {
        assert_eq!(p!(e!(x ^ 3), e!(x)).degree(), Some(3.into()));
        assert_eq!(
            p!(e!(3 * x ^ 2 + 4 * x + 5), e!(x)).degree(),
            Some(2.into())
        );
        assert_eq!(p!(e!(2 * x ^ 3), e!(x)).degree(), Some(3.into()));
        assert_eq!(p!(e!((x + 1) * (x + 3)), e!(x)).degree(), None);
        assert_eq!(
            p!(e!((x + 1) * (x + 3)).expand(), e!(x)).degree(),
            Some(2.into())
        );
        assert_eq!(p!(e!(3), e!(x)).degree(), Some(0.into()));

        assert_eq!(p!(e!(x ^ 3), e!(x)).degree(), Some(3.into()));
        assert_eq!(
            p!(e!(3 * x ^ 2 + 4 * x + 5), e!(x)).degree(),
            Some(2.into())
        );
        assert_eq!(p!(e!(2 * x ^ 3), e!(x)).degree(), Some(3.into()));
        assert_eq!(p!(e!((x + 1) * (x + 3)), e!(x)).degree(), None);
        assert_eq!(
            p!(e!((x + 1) * (x + 3)).expand(), e!(x)).degree(),
            Some(2.into())
        );
        assert_eq!(p!(e!(3), e!(x)).degree(), Some(0.into()));
        assert_eq!(
            p!(e!(3 * w * x ^ 2 * y ^ 3 * z ^ 4), [e!(x), e!(z)]).degree(),
            Some(6.into())
        );
        assert_eq!(
            p!(e!(a * x ^ 2 + b * x + c), e!(x)).degree(),
            Some(2.into())
        );
        assert_eq!(
            p!(e!(2 * x ^ 2 * y * z ^ 3 + w * x * z ^ 6), [e!(x), e!(z)]).degree(),
            Some(7.into())
        );
    }

    #[test]
    fn coeffs() {
        assert_eq!(
            p!(e!(a * x ^ 2 + b * x + c), e!(x)).coeffs_of(&e!(x)),
            [(2.into(), e!(a)), (1.into(), e!(b)), (0.into(), e!(c))]
                .into_iter()
                .collect()
        );
        assert_eq!(
            p!(e!(3 * x * y ^ 2 + 5 * x ^ 2 * y + 7 * x + 9), e!(x)).coeffs_of(&e!(x)),
            [
                (0.into(), Expr::from(9)),
                (1.into(), e!(7 + 3 * y ^ 2)),
                (2.into(), e!(5 * y))
            ]
            .into_iter()
            .collect()
        )
    }

    #[test]
    fn leading_coeff() {
        assert_eq!(
            p!(
                e!(3 * x * y ^ 2 + 5 * x ^ 2 * y + 7 * x ^ 2 * y ^ 3 + 9),
                e!(x)
            )
            .leading_coeff_of(&e!(x)),
            e!(5 * y + 7 * y ^ 3).into()
        )
    }

    #[test]
    fn basic() {
        let u = e!(a * (x ^ 2 + 1) ^ 2 + (x ^ 2 + 1));
        let vars = VarSet::from(e!(x ^ 2 + 1));
        let poly = u.as_polynomial(&vars);
        assert!(poly.check());
        assert_eq!(poly.degree(), Some(2.into()));
        assert_eq!(poly.coeffs_of_deg(&e!(x ^ 2 + 1), &Int::ONE), None);
        assert_eq!(
            poly.coeffs_of_deg(&e!(x ^ 2 + 1), &Int::ZERO),
            Some(e!(1 + x ^ 2))
        );
    }

    #[test]
    fn collect_terms() {
        let expr = e!(2*a*x*y + 3*b*x*y + 4*a*x + 5*b*x);
        assert_eq!(expr.as_polynomial(&[e!(x), e!(y)].into()).collect_terms().unwrap(), 
            e!((2*a + 3*b)*x*y + (4*a + 5*b)*x)
            )
    }
}
