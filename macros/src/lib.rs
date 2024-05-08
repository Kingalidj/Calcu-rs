use proc_macro2::TokenStream;
use quote::quote;
use syn::{punctuated as punc, Token};
use syn::parse;
use syn::parse::ParseStream;

#[derive(Debug, Clone, PartialEq)]
enum Condition {
    Any,
    Is(syn::Expr),
    IsNot(syn::Expr),
}

impl parse::Parse for Condition {
    fn parse(input: parse::ParseStream) -> syn::Result<Self> {

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

impl parse::Parse for CondTuple {
    fn parse(input: parse::ParseStream) -> parse::Result<Self> {
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

fn eval_bin(b: syn::ExprBinary) -> TokenStream {
    use syn::BinOp as B;

    let lhs = eval_expr(*b.left); 
    let rhs = eval_expr(*b.right);
    let op = match b.op {
        B::Add(_) => quote!(+),
        B::Sub(_) => quote!(-),
        B::Mul(_) => quote!(*),
        B::Div(_) => quote!(/),

        B::Eq(_) => quote!(==),
        B::Lt(_) => quote!(<),
        B::Le(_) => quote!(<=),
        B::Ne(_) => quote!(!=),
        B::Ge(_) => quote!(>=),
        B::Gt(_) => quote!(>),

        B::AddAssign(_) => quote!(+=),
        B::SubAssign(_) => quote!(-=),
        B::MulAssign(_) => quote!(*=),
        B::DivAssign(_) => quote!(/=),

        B::BitXor(_) => return quote!(#lhs.pow(#rhs)),
        _ => panic!("unknown binop"),
    };
    quote!(#lhs #op #rhs)
}

fn eval_lit(l: syn::ExprLit) -> TokenStream {
    match l.lit {
        syn::Lit::Int(i) => {
            let val: i64 = i.base10_parse().unwrap();
            quote!(calcu_rs::prelude::Expr::from(calcu_rs::prelude::Rational::from(#val)))
        },
        syn::Lit::Float(f) => {
            let val: f64 = f.base10_parse().unwrap();
            quote!(calcu_rs::prelude::Expr::from(calcu_rs::prelude::Float::new(#val)))
        },
        _ => panic!("unknown literal"),
    }
}

fn eval_paren(p: syn::ExprParen) -> TokenStream {
    let expr = eval_expr(*p.expr);
    quote!((#expr))
}

fn eval_unary(u: syn::ExprUnary) -> TokenStream {
    let expr = eval_expr(*u.expr);
    let op = u.op;
    quote!(#op #expr)
}

fn eval_ident(p: &syn::Ident) -> TokenStream {
    let mut id = p.to_string();
    if id == "oo" {
        quote!(calcu_rs::prelude::Expr::from(calcu_rs::prelude::Infinity::pos()))
    } else if id == "undef" {
        quote!(calcu_rs::prelude::Expr::from(calcu_rs::prelude::Numeric::Undefined))
    } else if id.starts_with('_') {
        let id = id.replacen('_', "?", 1);
        quote!(calcu_rs::prelude::Expr::PlaceHolder(#id))
    } else {
        quote!(calcu_rs::prelude::Expr::from(calcu_rs::prelude::Symbol::new(#id)))
    }
}

fn eval_path(p: syn::ExprPath) -> TokenStream {
    let id = p.path.get_ident().expect("path is not an identifier");
    eval_ident(id)
}

fn eval_expr(e: syn::Expr) -> TokenStream {
    use syn::Expr as E;
    match e {
        E::Binary(b) => eval_bin(b),
        E::Lit(l) => eval_lit(l),
        E::Paren(p) => eval_paren(p),
        E::Unary(u) => eval_unary(u),
        E::Path(p) => eval_path(p),
        _ => panic!("unknown expression: {:?}", e),
    }
}

#[proc_macro]
pub fn calc(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    eval_expr(syn::parse_macro_input!(input as syn::Expr)).into()
}
