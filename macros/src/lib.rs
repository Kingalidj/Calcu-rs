use proc_macro2::{Ident, Span, TokenStream, TokenTree};
use quote::{quote, ToTokens};
use syn::{
    parenthesized,
    parse::{self, discouraged::Speculative, Parse, ParseStream},
    parse_macro_input,
    punctuated::{self as punc, Punctuated},
    token, Attribute, Token,
};

mod rubi;

#[derive(PartialEq, Eq, Clone, Copy, Hash, PartialOrd, Ord)]
enum OpKind {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
}

impl std::fmt::Debug for OpKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let str = match self {
            OpKind::Add => "Add",
            OpKind::Sub => "Sub",
            OpKind::Mul => "Mul",
            OpKind::Div => "Div",
            OpKind::Pow => "Pow",
        };
        write!(f, "{}", str)
    }
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
        let (kind, span) = if let Ok(op) = input.parse::<Token![+]>() {
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
            return Err(parse::Error::new(
                input.span(),
                "expected operator { +, -, *, / }",
            ));
        };
        Ok(Self { kind, span })
    }
}

#[derive(Debug, PartialEq, Clone, PartialOrd)]
enum Expr {
    Num(i64),
    Float(f64),
    Symbol(String),
    Binary(OpKind, Box<Expr>, Box<Expr>),
    Func(syn::Ident, Vec<Expr>),
    Infinity { sign: i8 },
    Undef,
    PlaceHolder(String),
}

impl Expr {
    fn parse_operand(s: ParseStream) -> syn::Result<Expr> {
        if let Ok(id) = syn::Ident::parse(s) {
            let sid = id.to_string();
            if sid == "oo" {
                Ok(Expr::Infinity { sign: 1 })
            } else if sid == "undef" {
                Ok(Expr::Undef)
            } else if s.peek(token::Paren) {
                let content;
                let _: token::Paren = parenthesized!(content in s);
                let args: Punctuated<Expr, Token![,]> =
                    content.parse_terminated(Expr::parse, Token![,])?;
                Ok(Expr::Func(id, args.into_iter().collect()))
            } else {
                Ok(Expr::Symbol(sid.to_string()))
            }
        } else if let Ok(i) = syn::LitInt::parse(s) {
            let val: i64 = i.base10_parse().unwrap();
            Ok(Expr::Num(val))
        } else if let Ok(f) = syn::LitFloat::parse(s) {
            let val: f64 = f.base10_parse().unwrap();
            Ok(Expr::Float(val))
        } else if s.peek(token::Paren) {
            let content;
            parenthesized!(content in s);
            Expr::parse(&content)
        } else {
            Err(parse::Error::new(s.span(), "bad expression"))
        }
    }

    fn parse_unary_expr(s: ParseStream) -> syn::Result<Expr> {
        if let Ok(op) = Op::parse(s) {
            match op.kind {
                OpKind::Sub => {
                    let operand = Self::parse_operand(s)?;
                    Ok(if let Expr::Num(n) = operand {
                        Expr::Num(-1 * n)
                    } else {
                        Expr::Binary(OpKind::Mul, Expr::Num(-1).into(), operand.into())
                    })
                }
                _ => Err(parse::Error::new(op.span, "expected unary operator")),
            }
        } else if let Ok(_) = s.parse::<Token![?]>() {
            let mut id = "?".to_string();
            id.push_str(&syn::Ident::parse(s)?.to_string());
            Ok(Expr::PlaceHolder(id.to_string()))
        } else {
            Self::parse_operand(s)
        }
    }
    fn parse_bin_expr(s: ParseStream, prec_in: i32) -> syn::Result<Expr> {
        let mut expr = Self::parse_unary_expr(s)?;
        loop {
            if s.is_empty() {
                break;
            }

            if s.peek(Token![->]) || (s.peek(Token![<]) && s.peek2(Token![->])) || s.peek(Token![;])
            {
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
}

impl Parse for Expr {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        Expr::parse_bin_expr(input, 0 + 1)
    }
}

fn to_expr_stream(e: &Expr) -> parse::Result<TokenStream> {
    gen_expr_stream(e)
}

fn get_crate_name() -> TokenStream {
    quote!(calcu_rs)
}

fn gen_expr_stream(e: &Expr) -> parse::Result<TokenStream> {
    use Expr as E;
    use OpKind as OK;
    let cname = get_crate_name();
    Ok(match e {
        E::Num(n) => quote!(#cname::Expr::rational(#n)),
        E::Symbol(s) if s == "pi" => quote!(#cname::Expr::pi()),
        E::Symbol(s) => quote!(#cname::Expr::from(#s)),
        E::Undef => quote!(#cname::Expr::undef()),
        E::Binary(op, lhs, rhs) => {
            let lhs = gen_expr_stream(lhs)?;
            let rhs = gen_expr_stream(rhs)?;
            let op = match op {
                OK::Add => quote!(add),
                OK::Sub => quote!(sub),
                OK::Mul => quote!(mul),
                OK::Div => quote!(div),
                OK::Pow => quote!(pow),
            };
            quote! { #cname::Expr::#op(#lhs, #rhs)}
        }
        E::PlaceHolder(_) => {
            return Err(parse::Error::new(
                Span::call_site(),
                "placeholder not allowed in expressions, only in patterns",
            ))
        }
        E::Func(func, args) => {
            let mut args_tok = TokenStream::default();
            for a in args {
                let e = gen_expr_stream(a)?;
                args_tok.extend(quote!(#e, ));
            }
            quote!(#cname::Expr::#func(#args_tok))
        }
        _ => todo!(),
    })
}

#[derive(Debug, Clone, PartialEq)]
struct ExprArgs {
    //cntxt: Ident,
    expr: Expr,
}

impl Parse for ExprArgs {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        //let cntxt: Ident = input.parse()?;
        //let _: Token![:] = input.parse()?;
        let expr: Expr = input.parse()?;
        Ok(Self { expr })
    }
}

#[proc_macro]
pub fn expr(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let args = parse_macro_input!(input as ExprArgs);
    let stream = to_expr_stream(&args.expr);
    //panic!("{}", stream);
    match stream {
        Ok(s) => s.into(),
        Err(e) => e.to_compile_error().into(),
    }
}

#[proc_macro]
pub fn integration_rules(_: proc_macro::TokenStream) -> proc_macro::TokenStream {
    rubi::load_rubi();
    TokenStream::new().into()
}

struct ArithOpsArgs {
    ref_tok: syn::Token![ref],
    comma: syn::Token![,],
    field: syn::ExprField,
}

impl Parse for ArithOpsArgs {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        Ok(Self {
            ref_tok: input.parse()?,
            comma: input.parse()?,
            field: input.parse()?,
        })
    }
}

#[proc_macro_attribute]
pub fn arith_ops(
    attr: proc_macro::TokenStream,
    item: proc_macro::TokenStream,
) -> proc_macro::TokenStream {
    let item = parse_macro_input!(item as syn::ItemStruct);

    let mut attribs = quote! {
        #[derive(Add, AddAssign, Sub, SubAssign, Mul, MulAssign, Div, DivAssign)]
        #[mul(forward)]
        #[div(forward)]
        #[mul_assign(forward)]
        #[div_assign(forward)]
        #item
    };

    if !attr.is_empty() {
        let args: ArithOpsArgs = match syn::parse(attr) {
            Ok(id) => id,
            Err(e) => return e.into_compile_error().into(),
        };

        let member = args.field.member;

        #[allow(non_snake_case)]
        let T = item.ident;

        attribs.extend(quote! {
            impl ops::AddAssign<&#T> for #T {
                fn add_assign(&mut self, rhs: &Self) {
                    self.#member += &rhs.#member
                }
            }
            impl ops::Add<&#T> for #T {
                type Output = #T;
                fn add(self, rhs: &Self) -> Self::Output {
                    Self(self.#member + &rhs.#member)
                }
            }
            impl ops::SubAssign<&#T> for #T {
                fn sub_assign(&mut self, rhs: &Self) {
                    self.#member -= &rhs.#member
                }
            }
            impl ops::Sub<&#T> for #T {
                type Output = #T;
                fn sub(self, rhs: &Self) -> Self::Output {
                    Self(self.#member - &rhs.#member)
                }
            }
            impl ops::MulAssign<&#T> for #T {
                fn mul_assign(&mut self, rhs: &Self) {
                    self.#member *= &rhs.#member
                }
            }
            impl ops::Mul<&#T> for #T {
                type Output = #T;
                fn mul(self, rhs: &Self) -> Self::Output {
                    Self(self.#member * &rhs.#member)
                }
            }
            impl ops::Div<&#T> for #T {
                type Output = #T;
                fn div(self, rhs: &Self) -> Self::Output {
                    Self(self.#member / &rhs.#member)
                }
            }
            impl ops::DivAssign<&#T> for #T {
                fn div_assign(&mut self, rhs: &Self) {
                    self.#member /= &rhs.#member
                }
            }
        })
    }

    attribs.into()
}
