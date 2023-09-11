extern crate proc_macro as proc;

mod calcurs_scope;
mod macro_scope;
mod parsers;
mod utils;

use calcurs_scope::CalcursMacroScope;
use parsers::{FuncMacroArg, TraitArg};
use proc_macro2::{Span, TokenStream};
use quote::{quote, ToTokens};
use syn::{
    parse::{Parse, Parser},
    parse_macro_input, Error, Field, ItemStruct, ItemTrait, TraitBound,
};
use utils::*;

#[allow(unused_imports)]
// needed because of doclink bug
use calcurs_internals::Inherited;

/// This macro gets as input a type and will insert a field called base with that type it will also implement the [Inherited] trait
#[proc_macro_attribute]
pub fn inherit(attrib: proc::TokenStream, item: proc::TokenStream) -> proc::TokenStream {
    let mut item = parse_macro_input!(item as ItemStruct);
    let base_type = parse_macro_input!(attrib as syn::Type);

    let struct_name = &item.ident.clone();
    let struct_generics = &item.generics.clone();

    let base = Field::parse_named
        .parse2(quote! {base: #base_type})
        .expect("inherit: could not implemnt base field");

    let internals = import_crate("internals");

    let err = append_struct_field(&mut item, base);
    // let item = append_struct_field(item, base);

    if let Err(err) = err {
        return err.into_compile_error().into();
    }

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

/// Modifies the tokens based on the attributes used, e.g: [macro@calcurs_type] \
/// strips the module and inserts the modified [TokenStream]
///
/// a attribute macro + a mod was used, because: \
/// - function-like macros messed with the lsp \
/// - inner attribute macros are unstable \
/// - A build.rs file could be used to store it to the OUT_DIR and access those files in the
/// proc-macros
#[proc_macro_attribute]
pub fn init_calcurs_macro_scope(
    _: proc::TokenStream,
    item: proc::TokenStream,
) -> proc::TokenStream {
    let scope = parse_macro_input!(item as CalcursMacroScope);
    scope.into_token_stream().into()
}

/// struct marked with this attribute will be turned into CalcursTypes
///
/// this includes the following actions: \
/// add a base field of type defined by [macro@calcurs_base] and implement [Inherited]
#[proc_macro_attribute]
pub fn calcurs_type(_: proc::TokenStream, _: proc::TokenStream) -> proc::TokenStream {
    return Error::new(
        Span::call_site().into(),
        "called attribute calcurs_type outside a calcurs_scope",
    )
    .into_compile_error()
    .into();
}

/// traits marked with this attribute will be turned into CalcursTraits
#[proc_macro_attribute]
pub fn calcurs_trait(_: proc::TokenStream, _: proc::TokenStream) -> proc::TokenStream {
    return Error::new(
        Span::call_site().into(),
        "called attribute calcurs_trait outside a calcurs_scope",
    )
    .into_compile_error()
    .into();
}

/// can only be defined once per calcurs_macro_scope. the [macro@calcurs_type] macro will use the marked
/// struct as the base
#[proc_macro_attribute]
pub fn calcurs_base(_: proc::TokenStream, _: proc::TokenStream) -> proc::TokenStream {
    return Error::new(
        Span::call_site().into(),
        "called attribute calcurs_base outside a calcurs_scope",
    )
    .into_compile_error()
    .into();
}

fn impl_diff_debug_fn(base_type: &ItemStruct, default: &FuncMacroArg) -> syn::Result<TokenStream> {
    let fields = match &base_type.fields {
        syn::Fields::Named(syn::FieldsNamed { named, .. }) => {
            named.iter().map(|field| field.ident.clone().unwrap())
        }
        _ => {
            return Err(Error::new(
                base_type.ident.span(),
                "DiffDebug is not defined for structs without named fields",
            ))
        }
    };

    let ident = &base_type.ident;

    Ok(quote! {
        impl #ident {
            fn diff_debug(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                use std::fmt::Write;

                let default = #default();

                let mut diff = String::new();

                #(
                if (self.#fields != default.#fields) {
                    write!(diff, "{} = {:?}, ", stringify!(#fields), self.#fields)?;
                }
                )*

                write!(f, "Base {{{}}}", diff)
            }
        }
    })
}

#[proc_macro_attribute]
pub fn diff_debug(attrib: proc::TokenStream, input: proc::TokenStream) -> proc::TokenStream {
    let attrib = parse_macro_input!(attrib as FuncMacroArg);
    let input = parse_macro_input!(input as ItemStruct);

    match impl_diff_debug_fn(&input, &attrib) {
        Ok(stream) => stream.into(),
        Err(err) => err.to_compile_error().into(),
    }
}

#[proc_macro_attribute]
pub fn dyn_trait(attrib: proc::TokenStream, input: proc::TokenStream) -> proc::TokenStream {
    let mut input = parse_macro_input!(input as ItemTrait);

    // let attrib = match syn::Attribute::parse_outer.parse(attrib) {
    //     Ok(a) => a,
    //     Err(e) => return e.to_compile_error().into(),
    // };

    let args = match TraitArg::parse.parse(attrib) {
        Ok(a) => a,
        Err(e) => return e.to_compile_error().into(),
    };

    if args.path != "Into" {
        return syn::Error::new(args.path.span(), "Only Into trait supported as argument!")
            .to_compile_error()
            .into();
    }

    let into_type = &args.template_arg;

    let trait_name = &input.ident;
    let dyn_trait_name = quote::format_ident!("Dyn{}", trait_name);

    let doc = format!(
        "Allows for trait object: dyn [{}] to be cloned and compared.\n\n
Generated by [macros::dyn_trait]
        ",
        trait_name
    );

    let dyn_trait = quote! {

        #[doc = #doc]
        pub trait #dyn_trait_name {
            fn box_clone(&self) -> Box< dyn #trait_name >;
            fn typ_clone(&self) -> #into_type;
            fn as_any(&self) -> &dyn std::any::Any;
            fn as_obj(&self) -> &dyn #trait_name;
        }

        impl<T: #trait_name + Clone + PartialEq + 'static + #args> #dyn_trait_name for T {

            fn box_clone(&self) -> Box<dyn #trait_name> {
                Box::new(self.clone())
            }

            fn as_any(&self) -> &dyn std::any::Any {
                self
            }

            fn typ_clone(&self) -> #into_type {
                (*self).clone().into()
            }

            fn as_obj(&self) -> &dyn #trait_name {
                self
            }
        }

        impl Clone for Box<dyn #trait_name> {
            fn clone(&self) -> Box<dyn #trait_name> {
                #dyn_trait_name::box_clone(self.as_ref())
            }
        }
    };

    let trait_bound = syn::TypeParamBound::parse
        .parse2(quote! { #dyn_trait_name })
        .expect(&format!(
            "could not parse {} as trait bound",
            quote!(#dyn_trait_name)
        ));

    input.supertraits.push(trait_bound.into());

    let mut stream = TokenStream::new();
    dyn_trait.to_tokens(&mut stream);
    input.to_tokens(&mut stream);

    stream.into()
}
