use proc_macro as proc;
use proc_macro2::Span;
use quote::quote;
use syn::{
    parse::{Parse, ParseStream},
    parse_macro_input, FieldsUnnamed, Ident,
};

fn to_snake_case(str: &str) -> String {
    let mut buffer = String::with_capacity(str.len() + str.len() / 2);

    let mut str = str.chars();

    if let Some(first) = str.next() {
        let mut n2: Option<(bool, char)> = None;
        let mut n1: (bool, char) = (first.is_lowercase(), first);

        for c in str {
            let prev_n1 = n1.clone();

            let n3 = n2;
            n2 = Some(n1);
            n1 = (c.is_lowercase(), c);

            if let Some((false, c3)) = n3 {
                if let Some((false, c2)) = n2 {
                    if n1.0 && c3.is_uppercase() && c2.is_uppercase() {
                        buffer.push('_');
                    }
                }
            };

            buffer.push_str(&prev_n1.1.to_lowercase().to_string());

            if let Some((true, _)) = n2 {
                if n1.1.is_uppercase() {
                    buffer.push('_');
                }
            }
        }

        buffer.push_str(&n1.1.to_lowercase().to_string());
    }

    buffer
}

struct WrapperEnum {
    ident: Ident,
    variants: Vec<Ident>,
}

impl Parse for WrapperEnum {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let input = input.parse::<syn::ItemEnum>()?;

        let vars: Vec<_> = input
            .variants
            .into_iter()
            .filter(|v| match &v.fields {
                syn::Fields::Unnamed(FieldsUnnamed { unnamed, .. }) if unnamed.len() == 1 => true,
                _ => false,
            })
            .map(|var| var.ident)
            .collect();

        Ok(Self {
            ident: input.ident,
            variants: vars,
        })
    }
}

#[proc_macro_derive(Procagate)]
pub fn procagate(input: proc::TokenStream) -> proc::TokenStream {
    let input = parse_macro_input!(input as WrapperEnum);

    let macro_name = format!("procagate_{}", to_snake_case(&input.ident.to_string()));
    let macro_name = Ident::new(&macro_name, Span::call_site());

    let enum_name = input.ident;
    let vars = input.variants;

    let tok = quote! {
        macro_rules! #macro_name {
            ($self: ident, $v: ident => $bod: tt) => {
                match $self {
                    #(
                    #enum_name::#vars($v) => $bod
                    )*
                }
            };
        }
    };

    tok.into()
}
