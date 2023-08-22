use proc_macro2::TokenStream;
use quote::ToTokens;
use syn::parse_macro_input;

extern crate proc_macro as proc;

#[proc_macro_attribute]
pub fn init_calcrs_macro_scope(_: proc::TokenStream, item: proc::TokenStream) -> proc::TokenStream {
    let input: syn::ItemMod = parse_macro_input!(item as syn::ItemMod);

    let mut stream = TokenStream::new();

    input.content.map(|x| {
        x.1.into_iter()
            .for_each(|item: syn::Item| item.to_tokens(&mut stream))
    });

    stream.into()
}
