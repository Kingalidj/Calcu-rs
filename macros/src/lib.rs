// use proc_macro2::TokenStream;
// use quote::quote;
// use syn::parse_macro_input;

use proc_macro;
use quote::quote;
use syn::{
    parse::{Parse, Parser},
    parse_macro_input,
    punctuated::Punctuated,
    Field, FieldsNamed, Ident, Token,
};

use proc_macro2::{Span, TokenStream};
use proc_macro_crate::{crate_name, FoundCrate};

fn import_crate(name: &str) -> TokenStream {
    let found_crate = crate_name(name).expect(&format!("{name} is not present in Cargo.toml"));

    match found_crate {
        FoundCrate::Itself => quote!(crate),
        FoundCrate::Name(name) => {
            let ident = Ident::new(&name, Span::call_site());
            quote!( #ident )
        }
    }
}

/// this macro gets as input a type and will insert a field called base with that type it will also implement the [calcurs_internals::Inherited] trait
#[proc_macro_attribute]
pub fn inherit(
    attr: proc_macro::TokenStream,
    item: proc_macro::TokenStream,
) -> proc_macro::TokenStream {
    let mut item = parse_macro_input!(item as syn::ItemStruct);
    let base_type = parse_macro_input!(attr as syn::Type);

    let struct_name = &item.ident;
    let struct_generics = &item.generics;

    let base = Field::parse_named
        .parse2(quote! {base: #base_type})
        .unwrap();

    let internals = import_crate("internals");

    match item.fields {
        syn::Fields::Named(FieldsNamed { ref mut named, .. }) => named.push(base),
        _ => {
            return syn::Error::new_spanned(
                item.fields,
                "Only named fields are supported for adding the base field.",
            )
            .into_compile_error()
            .into()
        }
    };

    quote! {
        #item

        impl #struct_generics #internals::Inherited<#base_type> for #struct_name #struct_generics {
            fn base(&self)  -> &#base_type {
                &self.base
            }
        }

    }
    .into()
}

#[derive(Debug, Clone, PartialEq)]
enum IdentType {
    Trait(Ident),
    Field(Ident),
}

impl Parse for IdentType {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let ident: Ident = input.parse()?;

        if let Some(c) = ident.to_string().chars().next() {
            if c.is_uppercase() {
                Ok(IdentType::Trait(ident))
            } else {
                Ok(IdentType::Field(ident))
            }
        } else {
            panic!("identifier is empty");
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
struct Relation(IdentType, Vec<IdentType>);

impl Parse for Relation {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let lhs: IdentType = input.parse()?;
        if let IdentType::Field(ident) = lhs {
            return Err(syn::Error::new(
                ident.span(),
                "Only trait identifiers are allowed on the lhs",
            ));
        }

        let _: Token!(=>) = input.parse()?;
        let rhs: Punctuated<IdentType, Token!(&)> = Punctuated::parse_separated_nonempty(input)?;
        Ok(Relation(lhs, rhs.into_iter().collect()))
    }
}

#[derive(Debug, Clone, PartialEq, Default)]
struct CategoryRelations {
    relations: Vec<Relation>,
}

impl Parse for CategoryRelations {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let rels: Punctuated<Relation, Token!(,)> = Punctuated::parse_terminated(input)?;

        Ok(CategoryRelations {
            relations: rels.into_iter().collect(),
        })
    }
}

#[proc_macro]
pub fn define_categories(item: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let item = parse_macro_input!(item as CategoryRelations);

    let mut map: std::collections::HashSet<Ident> = Default::default();

    for rel in &item.relations {
        for field in &rel.1 {
            if let IdentType::Field(f) = field {
                map.insert(f.clone());
            }
        }
    }

    quote!().into()
}
