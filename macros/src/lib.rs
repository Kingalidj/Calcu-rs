use proc_macro2::{TokenStream, Span};
use quote::{quote, ToTokens};
use syn::{parse::{self, discouraged::Speculative, Parse, ParseStream}, punctuated as punc, spanned::Spanned, Token};
use std::fmt::Write;

#[derive(Debug, Clone, PartialEq)]
enum Condition {
    Any,
    Is(syn::Expr),
    IsNot(syn::Expr),
}

impl Parse for Condition {
    fn parse(input: ParseStream) -> syn::Result<Self> {

        let not = input.parse::<syn::Token![!]>().is_ok(); 

        let e: syn::Expr = input.parse()?;

        Ok(match e {
            syn::Expr::Infer(_) => Condition::Any,
            _ => match not {
                false => Condition::Is(e),
                true => Condition::IsNot(e),
            },
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
struct CondTuple {
    elems: Vec<Condition>,
}

impl Parse for CondTuple {
    fn parse(input: ParseStream) -> parse::Result<Self> {
        let tuple_bod;
        syn::parenthesized!(tuple_bod in input);

        let elems = punc::Punctuated::<Condition, syn::Token![,]>::parse_terminated(&tuple_bod)
            .unwrap()
            .into_iter()
            .collect();

        Ok(Self { elems })
    }
}


#[derive(Debug, Clone, PartialEq)]
struct Identity {
    is_default: bool,
    conditions: Vec<CondTuple>,
    equals: syn::Expr,
}

impl parse::Parse for Identity {
    fn parse(input: parse::ParseStream) -> syn::Result<Self> {
        if let syn::Result::Ok(_) = input.parse::<syn::Token![default]>() {
            let is_default = true;
            let _: syn::Token![=>] = input.parse()?;
            let equals = input.parse()?;
            let conditions = vec![];
            return Ok(Identity {
                is_default,
                conditions,
                equals,
            });
        }

        let is_default = false;

        let conditions = punc::Punctuated::<CondTuple, syn::Token![||]>::parse_separated_nonempty(input)
            .unwrap()
            .into_iter()
            .collect();

        let _: syn::Token![=>] = input.parse()?;

        let equals = input.parse()?;

        Ok(Identity {
            is_default,
            conditions,
            equals,
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
struct IdentityMap {
    match_expr: Vec<syn::Expr>,
    idents: Vec<Identity>,
}

impl parse::Parse for IdentityMap {
    fn parse(input: parse::ParseStream) -> syn::Result<Self> {
        let match_expr: syn::ExprTuple = input.parse()?;
        let match_expr: Vec<_> = match_expr.elems.into_iter().collect();

        let body;
        syn::braced!(body in input);

        let arms = punc::Punctuated::<Identity, syn::Token![,]>::parse_terminated(&body)?;
        let arms: Vec<_> = arms.into_iter().collect();

        Ok(Self {
            match_expr,
            idents: arms,
        })
    }
}

// identity!{ (a, b) {
//      (Zero, One) => ...
//      default => ...
// }}
//
// turns to:
//
// if a.is(Zero) && b.is(One) {
//
// } else if ...
//
// } else {
//      ...
// }

#[proc_macro]
pub fn identity(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let map: IdentityMap = syn::parse_macro_input!(input as IdentityMap);

    let n = map.match_expr.len();

    let mut out: TokenStream = Default::default();

    if map.idents.len() == 1 {
        let id = map.idents.get(0).unwrap();
        let equals = &id.equals;
        return quote!(#equals).into();
    }

    for id_indx in 0..map.idents.len() {
        let id = map.idents.get(id_indx).unwrap();

        if id.is_default {
            let equals = &id.equals;
            out.extend(quote!(else { #equals }));
            break;
        }

        out.extend(if id_indx == 0 {
            quote!(if)
        } else {
            quote!(else if)
        });


        for cond_indx in 0..id.conditions.len() {
            let cond = id.conditions.get(cond_indx).unwrap();

            assert_eq!(n, cond.elems.len());

            let mut cond_stream: TokenStream = Default::default();

            for i in 0..n {
                if i != 0 {
                    cond_stream.extend(quote!(&&));
                }

                let e = map.match_expr.get(i).unwrap();
                let cond = cond.elems.get(i).unwrap();

                let (c, not) = match cond {
                    Condition::Is(expr) => (quote!(#expr), quote!()),
                    Condition::IsNot(expr) => (quote!(#expr), quote!(!)),
                    Condition::Any => {
                        cond_stream.extend(quote!(true));
                        continue
                    },
                };

                cond_stream.extend(quote! {
                    #not #e.is(#c)
                });

            }

            out.extend(quote!((#cond_stream)));

            if cond_indx != id.conditions.len()-1 {
                out.extend(quote!(||));
            }
        }

        let equals = &id.equals;
        out.extend(quote!( { #equals } ));
    }

    out.into()
}


#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash, PartialOrd, Ord)]
enum OpKind {
    Add, Sub,
    Mul, Div,
    Pow,
}

#[derive(Debug, Clone, Copy)]
struct Op {
    kind: OpKind,
    span: Span,
}


impl OpKind {
   fn precedence(&self) -> i32 {
        match self {
            OpKind::Add | OpKind::Sub => 1,
            OpKind::Mul | OpKind::Div => 2,
            OpKind::Pow => 3,
        }
   }
}

impl Op {
    fn precedence(&self) -> i32 {
        self.kind.precedence()
    }
}

impl Parse for Op {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let (kind, span) =
            if let Ok(op) = input.parse::<Token![+]>() {
                (OpKind::Add, op.span)
            } else if let Ok(op) = input.parse::<Token![-]>() {
                (OpKind::Sub, op.span)
            } else if let Ok(op) = input.parse::<Token![*]>() {
                (OpKind::Mul, op.span)
            } else if let Ok(op) = input.parse::<Token![/]>() {
                (OpKind::Div, op.span)
            } else if let Ok(op) = input.parse::<Token![^]>() {
                (OpKind::Pow, op.span)
            } else {
                return Err(syn::parse::Error::new(input.span(), "expected operator { +, -, *, / }"));
            };
        Ok(Self {kind, span})
    }
}

#[derive(Debug, PartialEq, Clone, PartialOrd)]
enum Expr {
    Num(i32),
    Float(f32),
    Symbol(String),
    Binary(OpKind, Box<Expr>, Box<Expr>),
    Infinity{sign: i8},
    Undef,
    PlaceHolder(String),
}

impl Expr {
    fn parse_operand(s: ParseStream) -> syn::Result<Expr> {
        if let Ok(id) = syn::Ident::parse(s) {
            let id = id.to_string();
            if id == "oo" {
                Ok(Expr::Infinity { sign: 1 })
            } else if id == "undef" {
                Ok(Expr::Undef)
            } else {
                Ok(Expr::Symbol(id.to_string()))
            }

        } else if let Ok(i) = syn::LitInt::parse(s) {
            let val: i32 = i.base10_parse().unwrap();
            Ok(Expr::Num(val))

        } else if let Ok(f) = syn::LitFloat::parse(s) {
            let val: f32 = f.base10_parse().unwrap();
            Ok(Expr::Float(val))

        } else if s.peek(syn::token::Paren) {
            let content;
            syn::parenthesized!(content in s);
            Expr::parse(&content)
        } else {
            Err(syn::parse::Error::new(s.span(), "bad expression"))
        }
    }

    fn parse_unary_expr(s: ParseStream) -> syn::Result<Expr> {
        if let Ok(op) = Op::parse(s) {
            match op.kind {
                OpKind::Sub => {
                    let operand = Self::parse_operand(s)?;
                    Ok(Expr::Binary(OpKind::Mul, Expr::Num(-1).into(), operand.into()))
                }
                _ => Err(syn::parse::Error::new(op.span, "expected unary operator"))
            }
        }  else if let Ok(_) = s.parse::<Token![?]>() {
            let mut id = "?".to_string();
            id.push_str(&syn::Ident::parse(s)?.to_string());
            Ok(Expr::PlaceHolder(id.to_string()))
        } else {
            Self::parse_operand(s)
        }
    }
    fn parse_bin_expr(s: ParseStream, prec_in: i32) -> syn::Result<Expr> {
        let mut expr = Self::parse_unary_expr(s)?;
        loop
        {
            if s.is_empty() {
                break;
            }

            if s.peek(Token![->]) || (s.peek(Token![<]) && s.peek2(Token![->])) || s.peek(Token![;]) {
                break;
            }

            let ahead = s.fork();
            let op = match Op::parse(&ahead) {
                Ok(op) if op.precedence() < prec_in => break,
                Ok(op) => op,
                Err(_) => break,
            };

            s.advance_to(&ahead);

            let rhs = Expr::parse_bin_expr(s, op.precedence() + 1)?;
            expr = Expr::Binary(op.kind, expr.into(), rhs.into());
        }

        Ok(expr)
    }

    fn eval_op(op: OpKind, lhs: TokenStream, rhs: TokenStream) -> TokenStream {
        match op {
            OpKind::Add => quote!((#lhs + #rhs)),
            OpKind::Sub => quote!((#lhs - #rhs)),
            OpKind::Mul => quote!((#lhs * #rhs)),
            OpKind::Div => quote!((#lhs / #rhs)),
            OpKind::Pow => quote!((#lhs.pow(#rhs))),
        }
    }

    fn quote(&self) -> TokenStream {
        match self {
            Expr::Num(v) =>
                quote!(::calcu_rs::prelude::Expr::from(::calcu_rs::prelude::Rational::from(#v))),
            Expr::Float(v) =>
                quote!(::calcu_rs::prelude::Expr::from(::calcu_rs::prelude::Float::from(#v))),
            Expr::Symbol(s) =>
                quote!(::calcu_rs::prelude::Expr::from(::calcu_rs::prelude::Symbol::new(#s))),
                Expr::Binary(op, l, r) => {
                    let lhs = l.quote();
                    let rhs = r.quote();
                    Self::eval_op(*op, lhs, rhs)
                }
            Expr::Infinity { sign } => {
                if sign.is_negative() {
                    quote!(::calcu_rs::prelude::Expr::from(::calcu_rs::prelude::Infinity::neg()))
                } else {
                    quote!(::calcu_rs::prelude::Expr::from(::calcu_rs::prelude::Infinity::pos()))
                }
            }
            Expr::Undef => {
                quote!(::calcu_rs::prelude::Expr::Undefined)
            }
            Expr::PlaceHolder(s) => {
                quote!(::calcu_rs::prelude::Expr::PlaceHolder(#s))
            }
        }
    }

}

impl syn::parse::Parse for Expr {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        Expr::parse_bin_expr(input, 0 + 1)
    }
}

//fn eval_expr(expr: &Expr) -> TokenStream {
//    match expr {
//        Expr::Num(v) =>
//            quote!(::calcu_rs::prelude::Expr::from(::calcu_rs::prelude::Rational::from(#v))),
//        Expr::Float(v) =>
//            quote!(::calcu_rs::prelude::Expr::from(::calcu_rs::prelude::Float::from(#v))),
//        Expr::Symbol(s) =>
//            quote!(::calcu_rs::prelude::Expr::from(::calcu_rs::prelude::Symbol::new(#s))),
//        Expr::Binary(op, l, r) => {
//            let lhs = eval_expr(l);
//            let rhs = eval_expr(r);
//            eval_op(*op, lhs, rhs)
//        }
//        Expr::Infinity { sign } => {
//            if sign.is_negative() {
//                quote!(::calcu_rs::prelude::Expr::from(::calcu_rs::prelude::Infinity::neg()))
//            } else {
//                quote!(::calcu_rs::prelude::Expr::from(::calcu_rs::prelude::Infinity::pos()))
//            }
//        }
//        Expr::Undef => {
//            quote!(::calcu_rs::prelude::Expr::Undefined)
//        }
//        Expr::PlaceHolder(s) => {
//            quote!(::calcu_rs::prelude::Expr::PlaceHolder(#s))
//        }
//    }
//}

#[proc_macro]
pub fn calc(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    syn::parse_macro_input!(input as Expr).quote().into()
}

#[derive(Debug, Clone)]
struct RewriteRule {
    name: String,
    lhs: Expr,
    rhs: Expr,
    cond: Option<syn::Expr>,
    bidir: bool,
}

impl RewriteRule {

    fn quote_lhs_to_rhs(name: &String, lhs: &Expr, rhs: &Expr, cond: &Option<syn::Expr>, dbg: bool) -> TokenStream {
        let lhs = lhs.quote();
        let rhs = rhs.quote();

        let mut debug = TokenStream::new();
        if dbg {
            let cond_str =
            match cond {
                Some(cond) => {
                    let mut str = " if ".to_string();
                    write!(str, "{},", cond.clone().to_token_stream().to_string()).unwrap();
                    str
                },
                None => ",".into(),
            };

            debug = quote!(
                println!("  {}: {} => {}{}", #name, __searcher, __applier, #cond_str);
            )
        }

        let mut cond_applier = TokenStream::new();

        if let Some(cond) = cond {
            cond_applier = quote!(
                let __applier = ::egg::ConditionalApplier {
                    condition: #cond,
                    applier: __applier,
                };
            )
        }

        quote!({
            let __searcher = ::egg::Pattern::from(&#lhs);
            let __applier  = ::egg::Pattern::from(&#rhs);
            #debug
            #cond_applier
            ::egg::Rewrite::new(#name.to_string(), __searcher, __applier).unwrap()
        })
    }

    fn quote_debug(&self, dbg: bool) -> TokenStream {
        if self.bidir {
            let n1 = self.name.clone();
            let mut n2 = self.name.clone();
            n2.push_str(" REV");
            let r1 = Self::quote_lhs_to_rhs(&n1, &self.lhs, &self.rhs, &self.cond, dbg);
            let r2 = Self::quote_lhs_to_rhs(&n2, &self.rhs, &self.lhs, &self.cond, dbg);
            quote!(#r1, #r2)
        } else {
            Self::quote_lhs_to_rhs(&self.name, &self.lhs, &self.rhs, &self.cond, dbg)
        }
    }

    fn quote(&self) -> TokenStream {
        self.quote_debug(false)
    }
}

impl Parse for RewriteRule {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let mut name = syn::Ident::parse(input)?.to_string();

        loop {
            if let Ok(n) = syn::Ident::parse(input) {
                name.push_str(" ");
                name.push_str(&n.to_string());
            } else if let Ok(n) = syn::Lit::parse(input) {
                name.push_str(" ");
                name.push_str(&n.to_token_stream().to_string());
            } else {
                break;
            }
        }

        let _ = input.parse::<Token![:]>()?;

        let lhs = Expr::parse(input)?;

        let bidir = 
            if input.peek(Token![->]) {
                let _ = input.parse::<Token![->]>()?;    
                false
            } else if input.peek(Token![<]) && input.peek2(Token![->]) {
                let _ = input.parse::<Token![<]>()?;    
                let _ = input.parse::<Token![->]>()?;    
                true
            } else {
                return Err(syn::parse::Error::new(input.span(), "expected -> or <->"));
            };

        let rhs = Expr::parse(input)?;

        let cond =
        if let Ok(_) = input.parse::<Token![if]>() {
            Some(syn::Expr::parse(input)?)
        } else {
            None
        };

        Ok(RewriteRule { name, lhs, rhs, cond, bidir })
    }
}

#[derive(Debug, Clone)]
struct RuleSet {
    gen_name: syn::Ident,
    rules: Vec<RewriteRule>,
    debug: bool,
}

impl RuleSet {
    fn quote(&self) -> TokenStream {
        let gen_name = &self.gen_name;

        let mut n: usize = 0;
        for r in &self.rules {
            n += 
                if r.bidir {
                    2
                } else {
                    1
                };
        }

        let mut rules = TokenStream::new();
        for r in &self.rules {
            let r = r.quote_debug(self.debug);
            rules.extend(quote!(#r,))
        }

        let mut debug = TokenStream::new();
        if self.debug {
            let name = gen_name.to_string();
            debug = quote!(println!("{}:", #name););
        }

        quote!(
            pub fn #gen_name() -> [::egg::Rewrite<Self, <Self as GraphExpression>::Analyser>; #n] {
                #debug
                [ #rules ]
            }
        )
    }
}

impl Parse for RuleSet {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let mut gen_name = syn::Ident::parse(input)?;
        let mut debug = false;

        if gen_name == "debug" {
            debug = true;
            gen_name = syn::Ident::parse(input)?;
        }

        let _ = input.parse::<Token![:]>();
        let rules: Vec<_> = punc::Punctuated::<RewriteRule, syn::Token![,]>::parse_terminated(&input)?.
            into_iter().collect();

        Ok(RuleSet { gen_name, rules, debug })
    }
}

#[proc_macro]
pub fn define_rules(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    syn::parse_macro_input!(input as RuleSet).quote().into()
}
