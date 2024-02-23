use proc_macro2::TokenStream;
use quote::quote;
use syn::punctuated as punc;
use syn::parse;

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
// to:
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

#[proc_macro]
pub fn calc(input: proc_macro::TokenStream) -> proc_macro::TokenStream {

    let expr: syn::Expr = syn::parse_macro_input!(input);

    Default::default()
}
