use crate::atom::{Atom, Expr, Real};
use crate::rational::Rational;

//calcurs_macros::integration_rules!();

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Hash)]
pub(crate) enum WlfrmAtom {
    None,
    Bool(bool),
    Expr(Expr),
}

impl WlfrmAtom {
    const FALSE: Self = WlfrmAtom::Bool(false);
    const TRUE: Self = WlfrmAtom::Bool(true);
}

impl From<bool> for WlfrmAtom {
    fn from(value: bool) -> Self {
        WlfrmAtom::Bool(value)
    }
}
impl From<Expr> for WlfrmAtom {
    fn from(value: Expr) -> Self {
        WlfrmAtom::Expr(value)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Hash)]
pub(crate) enum WlfrmFuncArgs {
    One(WlfrmAtom),
    Two([WlfrmAtom; 2]),
    Three([WlfrmAtom; 3]),
    Var(Vec<WlfrmAtom>),
}

impl WlfrmFuncArgs {
    fn len(&self) -> usize {
        match self {
            WlfrmFuncArgs::One(_) => 1,
            WlfrmFuncArgs::Two(_) => 2,
            WlfrmFuncArgs::Three(_) => 3,
            WlfrmFuncArgs::Var(vec) => vec.len(),
        }
    }

    fn get_arg(self) -> Option<WlfrmAtom> {
        match self {
            WlfrmFuncArgs::One(arg) => Some(arg),
            _ => None,
        }
    }

    fn get_two_args(self) -> Option<[WlfrmAtom; 2]> {
        match self {
            WlfrmFuncArgs::Two(args) => Some(args),
            _ => None,
        }
    }
}

macro_rules! wlfrm_fn {
    (func_body: ) => {
        { todo!() }
    };
    (func_body: $block:expr) => {
        $block
    };
    //($name:ident $(()$block:expr)? $(, $names: ident $($blocks:expr)?)* $(,)?) => {
    //    pub(crate) fn $name(arg: WlfrmExpr) -> WlfrmExpr { wlfrm_fn!{ func_body: $($block)? } }
    //    $(pub(crate) fn $names(arg: WlfrmExpr) -> WlfrmExpr { wlfrm_fn! { func_body: $($blocks)? }})*
    //}
    ($name:ident $arg:tt { $($body:tt)* } $($names:ident $args:tt { $($bodies:tt)* })*) => {
        pub(crate) fn $name $arg -> WlfrmAtom { {$($body)*}.into() }
        $(pub(crate) fn $names $args -> WlfrmAtom { {$($bodies)*}.into() })*
    }
}

macro_rules! error_msg {
    ($($tt:tt)*) => {
        panic!($($tt)*)
    }
}

macro_rules! expr {
    ($e:expr) => {{
        match $e {
            WlfrmAtom::Expr(e) => e,
            e => error_msg!("expected a symbolic expression, found {:?}", e),
        }
    }};
    ($e:expr $(, $es:expr)+) => {{
        [expr!($e) $(, expr!($es))+]
    }}
}

macro_rules! args {
    ($args:expr, 1) => {{
        match $args {
            WArgs::One(arg) => arg,
            args => error_msg!("expected one argument, found: {:?}", args),
        }
    }};
    ($args:expr, 2) => {{
        match $args {
            WArgs::Two(args) => args,
            args => error_msg!("expected two argument, found: {:?}", args),
        }
    }};
}

macro_rules! arg {
    ($args:expr) => {
        args!($args, 1)
    };
}

pub(crate) struct WlfrmBuiltins;

use WlfrmAtom as WA;
use WlfrmFuncArgs as WArgs;

#[allow(non_snake_case, dead_code, unused_variables)]
impl WlfrmBuiltins {
    wlfrm_fn! {

    FalseQ(arg: WArgs) {
        arg!(arg) == WA::FALSE
    }

    TrueQ(arg: WArgs) {
        arg!(arg) == WA::FALSE
    }

    IntegerQ(arg: WArgs) {
        match arg!(arg) {
            WA::Expr(e) => e.is_int(),
            _ => false
        }
    }

    FractionQ(arg: WArgs) {
        match arg!(arg) {
            WA::Expr(e) => e.is_rational_and(Rational::is_fraction),
            _ => false
        }
    }

    NumberQ(arg: WArgs) {
        match arg!(arg) {
            WA::Expr(e) => e.is_number(),
            _ => false,
        }
    }

    EvenQ(arg: WArgs) {
        match arg!(arg) {
            WA::Expr(e) => e.is_even(),
            _ => false
        }
    }

    OddQ(arg: WArgs) {
        match arg!(arg) {
            WA::Expr(e) => e.is_odd(),
            _ => false
        }
    }

    ProductQ(arg: WArgs) {
        match arg!(arg) {
            WA::Expr(e) => e.is_prod(),
            _ => false
        }
    }

    SumQ(arg: WArgs) {
        match arg!(arg) {
            WA::Expr(e) => e.is_sum(),
            _ => false
        }
    }

    PowQ(arg: WArgs) {
        match arg!(arg) {
            WA::Expr(e) => e.is_pow(),
            _ => false
        }
    }

    // yields True if expr is an expression which cannot be divided into subexpressions, and yields False otherwise.
    AtomQ(arg: WArgs) {
        match arg!(arg) {
            WA::Expr(e) => e.is_irreducible(),
            _ => false
        }
    }

    ComplexNumberQ(_: WArgs) { false }

    //PossibleZeroQ,
    //NonsumQ,
    //OrderedQ,
    //CoprimeQ,
    //PerfectSquareQ,
    //MemberQ,
    //FreeQ,

    //QuadraticProductQ,
    //PowerQ,
    //PowerOfLinearQ,
    //PowerOfLinearMatchQ,
    //IntegerPowerQ,
    //FractionalPowerQ,
    //IntegersQ,
    //PolynomialQ,
    //ComplexNumberQ,
    //NumericQ,
    //ListQ,

    Sin(arg: WArgs) {
        let a = expr!(arg!(arg));
        Expr::sin(a)
    }
    ArcSin(arg: WArgs) {
        let a = expr!(arg!(arg));
        Expr::arc_sin(a)
    }
    Cos(arg: WArgs) {
        let a = expr!(arg!(arg));
        Expr::cos(a)
    }
    ArcCos(arg: WArgs) {
        let a = expr!(arg!(arg));
        Expr::arc_cos(a)
    }
    Tan(arg: WArgs) {
        let a = expr!(arg!(arg));
        Expr::tan(a)
    }
    ArcTan(arg: WArgs) {
        let a = expr!(arg!(arg));
        Expr::arc_tan(a)
    }

    //Sinh,
    //ArcSinh,
    //Sec,
    //sec,
    //ArcSec,
    //Sech,
    //ArcSech,
    //Sinc,

    //Cos,
    //cos,
    //ArcCos,
    //Cosh,
    //ArcCosh,
    //Csc,
    //csc,
    //ArcCsc,
    //Csch,
    //ArcCsch,
    //Cot,
    //cot,
    //ArcCot,
    //Coth,
    //ArcCoth,

    //tan,
    //Tan,
    //ArcTan,
    //Tanh,
    //ArcTanh,

    Sqrt(arg: WArgs) {
        let e = expr!(arg!(arg));
        Expr::sqrt(e)
    }

    Log(args: WArgs) {
        if args.len() == 1 {
            let e = expr!(arg!(args));
            Expr::ln(e)
        } else if args.len() == 2 {
            let [ab, ae] = args!(args, 2);
            let [b, e] = expr!(ab, ae);

            let base: Real = match b.atom() {
                Atom::Rational(r) => r.clone().into(),
                Atom::Irrational(ir) => (*ir).into(),
                b => error_msg!("expected real log base, found: {b:?}"),
            };
            Expr::log(base, e)
        } else {
            error_msg!("expected one or two arguments, found: {:?}", args)
        }
    }

    Exp(arg: WArgs) {
        let e = expr!(arg!(arg));
        Expr::exp(e)
    }
    //LogGamma,
    //LogIntegral,
    //ProductLog,
    //PolyLog,


    //Binomial,

    //Sign,
    //Denominator,
    //Numerator ,
    //Exponent,
    //Quotient,
    //FractionalPart,
    //PolynomialRemainder,
    //PolynomialQuotient,
    //PolynomialQuotientRemainder,
    //IntegerPart,
    //Coefficient,
    //CoefficientList,
    //Last,
    //Head,

    //Sum,
    //Product,
    //Mod,
    //Max,
    //GCD,

    //Floor,
    //Abs,
    Not(arg: WArgs) {
        let e = arg!(arg);
        match e {
            WlfrmAtom::Bool(b) => !b,
            _ => todo!(),
            }
    }
    //Zeta,

    //Factor,
    //FactorInteger,
    //FactorSquareFreeList,

    Derivative(args: WArgs) {
        let [f, x] = args!(args, 2);
        let [f, x] = expr!(f, x);
        f.derivative(x)
    }
    //SinhIntegral,
    //SinIntegral,
    //CoshIntegral,
    //CosIntegral,
    //ExpIntegralE,
    //Integral,

    //Simplify,
    //ReplaceAll,
    //ReplacePart,
    //TrigExpand,
    //Flatten,
    Expand(arg: WArgs) {
        let e = expr!(arg!(arg));
        e.expand()
    }
    //FunctionExpand,
    //TrigToExp,
    //FullSimplify,
    //NormalizePowerOfLinear,
    //TrigReduce,
    //Map,

    //If,
    //Switch,
    //While,
    //Print,
    //Message,
    //Quiet,

    //H_,
    //Clear,
    //Apart,
    //Erfc,
    //Select,
    //f,
    //Reverse,
    //Sow,
    //Together,
    //Rule,
    //Scan,
    //Rest,
    //Distrib,
    //func,
    //Defer,
    //Dist,
    //G,
    //Erfi,
    //CannotIntegrate,
    //DeleteCases,
    //FresnelS,
    //EllipticE,
    //Cancel,
    //ExpIntegralEi,
    //AppendTo,
    //CheckArguments,
    //Function,
    //SetAttributes,
    //g_,
    //Identity,
    //Return,
    //First,
    //Apply,
    //f_,
    //Distribute,
    //J,
    //Hypergeometric2F1,
    //EllipticPi,
    //ShowStep,
    //Catch,
    //Discriminant,
    //Gamma,
    //H,
    //trig,
    //func_,
    //Hold,
    //Unintegrable,
    //Append,
    //Unique,
    //N,
    //F,
    //Throw,
    //G_,
    //Re,
    //Root,
    //Complex,
    //PolyGamma,
    //EllipticF,
    //RuleDelayed,
    //FresnelC,
    //AppellF1,
    //trig_,
    //BesselJ,
    //J_,
    //Im,
    //InverseFunction,
    //LeafCount,
    //Order,
    //ClearAll,
    //Unevaluated,
    //HypergeometricPFQ,
    //Refine,
    //Reap,
    //F_,
    //Do,
    //Min,
    //PolynomialGCD,
    //LCM,
    //Erf,
    //h,
    //D,
    //g,
    //Prepend,
    //Sort,
    //Drop,
    //Length,
    //DownValues,
    //TimeConstrained,
    }
}
