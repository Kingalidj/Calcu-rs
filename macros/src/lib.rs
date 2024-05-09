use proc_macro2::{TokenStream, Span};
use quote::quote;
use syn::{punctuated as punc, Token, parse::{ParseStream, Parse, self}, spanned::Spanned};

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
    fn parse_operand(s: &mut ParseStream) -> syn::Result<Expr> {
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
            let err_expr = syn::Expr::parse(s)?;
            Err(syn::parse::Error::new(err_expr.span(), "bad expression"))
        }
    }

    fn parse_unary_expr(s: &mut ParseStream) -> syn::Result<Expr> {
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
    fn parse_bin_expr(s: &mut ParseStream, prec_in: i32) -> syn::Result<Expr> {
        let mut expr = Self::parse_unary_expr(s)?;
        loop
        {
            if s.is_empty() {
                break;
            }

            {
                let tmp_s = s.fork();
                let op = Op::parse(&tmp_s)?;
                let op_prec = op.precedence();
                if op_prec < prec_in {
                    break;
                }
            }

            let op = Op::parse(s)?;
            let op_prec = op.precedence();

            let rhs = Expr::parse_bin_expr(s, op_prec + 1)?;
            expr = Expr::Binary(op.kind, expr.into(), rhs.into());
        }

        Ok(expr)
    }
}

impl syn::parse::Parse for Expr {
    fn parse(mut input: ParseStream) -> syn::Result<Self> {
        Expr::parse_bin_expr(&mut input, 0 + 1)
    }
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

fn eval_expr2(expr: &Expr) -> TokenStream {
    match expr {
        Expr::Num(v) =>
            quote!(::calcu_rs::prelude::Expr::from(::calcu_rs::prelude::Rational::from(#v))),
        Expr::Float(v) =>
            quote!(::calcu_rs::prelude::Expr::from(::calcu_rs::prelude::Float::from(#v))),
        Expr::Symbol(s) =>
            quote!(::calcu_rs::prelude::Expr::from(::calcu_rs::prelude::Symbol::new(#s))),
        Expr::Binary(op, l, r) => {
            let lhs = eval_expr2(l);
            let rhs = eval_expr2(r);
            eval_op(*op, lhs, rhs)
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

#[proc_macro]
pub fn calc(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    //eval_expr(syn::parse_macro_input!(input as syn::Expr)).into()
    eval_expr2(&syn::parse_macro_input!(input as Expr)).into()
}
