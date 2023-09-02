use proc_macro2::{Ident, TokenStream};
use quote::{quote, ToTokens};
use syn::parse::{Parse, ParseStream};
use syn::Token;

pub struct FuncMacroArg {
    pub is_const: bool,
    pub path: syn::Path,
}

impl Parse for FuncMacroArg {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let r#const: syn::Result<Token![const]> = input.parse();
        let is_const = r#const.is_ok();
        let path = syn::Path::parse(input)?;
        Ok(Self { is_const, path })
    }
}

impl ToTokens for FuncMacroArg {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        let path = &self.path;

        tokens.extend(quote! {
            #path
        });
    }
}

#[derive(Debug, Clone)]
pub struct FieldAssign {
    pub field: Ident,
    pub value: syn::ExprLit,
}

impl Parse for FieldAssign {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let field: Ident = input.parse()?;
        let _: syn::token::Eq = input.parse()?;
        let value: syn::ExprLit = input.parse()?;

        Ok(Self { field, value })
    }
}

pub type FieldValues = syn::punctuated::Punctuated<FieldAssign, Token!(,)>;
