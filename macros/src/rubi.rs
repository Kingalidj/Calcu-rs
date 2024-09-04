use std::{
    collections::{HashMap, HashSet},
    fmt::{self, Write},
    ops,
};

use logos::{Logos, Source};
use ordered_float::OrderedFloat;
use serde::{Deserialize, Serialize};

type F64 = OrderedFloat<f64>;

const INTEGRATION_RULES: &'static str = include_str!("test_rubi.m");

macro_rules! error {
    ($($tt:tt)*) => {{
        use std::fmt::Write;
        let mut buf = String::new();
        write!(buf, "ERROR: file: {}, line: {}\n", file!(), line!()).unwrap();
        write!(buf, $($tt)*).unwrap();
        panic!("{}", buf);
    }}
}

macro_rules! function_name {
    ($lvl: literal) => {{
        fn __f__() {}
        fn type_name_of<T>(_: T) -> &'static str {
            std::any::type_name::<T>()
        }
        type_name_of(__f__)
            .split("::")
            .skip($lvl)
            .filter(|&name| name != "__f__")
            .collect::<Vec<_>>()
            .join("::")

        //.find(|&part| part!= "f" && part != "{{closure}}")
        //.expect("function name")
    }};
}

macro_rules! trace {
    () => {{
        use std::fmt::Write;
        let mut buff = String::new();
        write!(buff, "TRACE: {}", function_name!(2)).unwrap();
        println!("{}", buff);
    }};
    ($($tt:tt)+) => {{
        use std::fmt::Write;
        let mut buff = String::new();
        write!(buff, "TRACE: {}: ", function_name!(2)).unwrap();
        write!(buff, $($tt)*).unwrap();
        println!("{}", buff);
    }}
}

fn parse_string(lex: &mut logos::Lexer<Wolfram>) -> String {
    let chars = lex.remainder().chars();

    let mut str = String::new();
    let mut escape = false;
    let mut len = 0;

    for c in chars {
        len += 1;

        if escape {
            match c {
                'n' => str.push('\n'),
                't' => str.push('\t'),
                _ => str.push_str(&format!("\\{}", c)),
            }
            escape = false;
        } else {
            match c {
                '\\' => escape = true,
                '"' => break,
                _ => str.push(c),
            }
        }
    }

    lex.bump(len);
    str
}

fn parse_comment(lex: &mut logos::Lexer<Wolfram>) -> String {
    let mut res = String::new();
    loop {
        let tok = lex.next().expect("unclosed comment");
        res += lex.slice();
        if tok == Ok(Wolfram::CommentEnd) {
            break;
        }
    }
    res
    //let end = lex.spanned().position(|(t, _)| t == Ok(Wolfram::CommentEnd)).expect("comment was not closed");
    //let mut res = String::new();
    //for _ in 0..end {
    //    res += lex.slice();
    //    println!("{}", res);
    //    lex.next().unwrap();
    //}
    //panic!("{}", end);
    //res
}

#[derive(Logos, Clone, Debug, PartialEq, PartialOrd)]
#[logos(skip r"[ \t\r\f]+")]
enum Wolfram {
    #[token("(*", parse_comment)]
    Comment(String),
    #[token("*)")]
    CommentEnd,
    #[token("\n")]
    NL,

    //#[regex("[_a-zA-Z][_0-9a-zA-Z]*", |lex| lex.slice().to_owned())]
    #[regex("[A-Za-z$_][A-Za-z0-9$_]*", |lex| lex.slice().to_owned())]
    Ident(String),

    #[regex(r"[0-9]+\.", |lex| lex.slice().parse::<f64>().unwrap() as u64, priority=100)]
    #[regex(r"[0-9]+", |lex| lex.slice().parse::<u64>().unwrap(), priority=100)]
    Integer(u64),

    #[regex(r"(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?", |lex| lex.slice().parse::<f64>().unwrap(), priority=5)]
    Float(f64),

    #[token("\"", parse_string)]
    Str(String),

    #[token("+")]
    Add,
    #[token("-")]
    Sub,
    #[token("*")]
    #[token("\\[Star]")]
    Mul,
    #[token("/")]
    Div,
    #[token("^")]
    Pow,

    #[token("&&")]
    And,
    #[token("||")]
    Or,
    #[token("!")]
    Not,

    #[token("==")]
    Eq,
    #[token("!=")]
    NEq,
    #[token("<")]
    Lt,
    #[token(">")]
    Ge,
    #[token("<=")]
    LtEq,
    #[token(">=")]
    GeEq,
    #[token("===")]
    Same,
    #[token("=!=")]
    UnSame,

    #[token("=")]
    Assign,
    #[token(":=")]
    Def,
    #[token("/;")]
    Cond,
    #[token("/.")]
    Replace,
    #[token("->")]
    Rule,
    #[token("\\")]
    BackSlash,
    #[token("#")]
    Slot,
    #[token("'")]
    Deriv,

    #[token("(")]
    LParen,
    #[token(")")]
    RParen,
    #[token("[")]
    LBracket,
    #[token("]")]
    RBracket,
    #[token("{")]
    LBrace,
    #[token("}")]
    RBrace,

    #[token(".")]
    Dot,
    #[token(",")]
    Comma,
    #[token(";")]
    Semicolon,
    #[token(":")]
    Colon,
    #[token("::")]
    DoublColon,
}

impl Wolfram {
    const fn op_prec(&self) -> u32 {
        use Wolfram as W;
        match self {
            W::Replace => 1,
            W::Rule => 2,
            W::Assign => 3,
            W::Or => 4,
            W::And => 5,
            W::NEq | W::Eq | W::Ge | W::GeEq | W::Lt | W::LtEq | W::Same | W::UnSame => 6,
            W::Add | W::Sub => 7,
            W::Mul | W::Div => 8,
            W::Pow => 9,
            _ => 0,
        }
    }

    const fn is_op(&self) -> bool {
        self.op_prec() != 0
    }

    const fn skippable(&self) -> bool {
        use Wolfram as W;
        match self {
            W::Comment(_) | W::CommentEnd | W::NL => true,
            _ => false,
        }
    }
}

#[derive(Clone, PartialEq, Eq, Hash, Deserialize, Serialize)]
struct Pattern {
    pat: Box<SymExpr>,
    cond: Option<Box<SymExpr>>,
}

impl fmt::Debug for Pattern {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(_) = &self.cond {
            write!(f, "Cond[")?;
        }
        write!(f, "{:?}", self.pat)?;
        if let Some(cond) = &self.cond {
            write!(f, " , {cond:?}]")?;
        }
        Ok(())
    }
}

#[derive(Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
struct FuncDef {
    call: FuncCall,
    pat: Pattern,
    //rhs: Box<SymExpr>,
    //cond: Option<Box<SymExpr>>,
}

impl FuncDef {
    fn for_each<F>(&self, f: F)
    where
        F: Fn(&SymExpr) + Copy,
    {
        self.call.for_each(f);
        self.pat.for_each(f);
    }
}

impl FuncCall {
    fn for_each<F>(&self, f: F)
    where
        F: Fn(&SymExpr) + Copy,
    {
        self.args.iter().for_each(|e| e.for_each(f));
    }
}

impl Pattern {
    fn for_each<F>(&self, f: F)
    where
        F: Fn(&SymExpr) + Copy,
    {
        self.pat.for_each(f);
        if let Some(cond) = &self.cond {
            cond.for_each(f)
        }
    }
}

impl fmt::Debug for FuncDef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?} := {:?}", self.call, self.pat)?;
        Ok(())
    }
}

#[derive(Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
struct FuncCall {
    name: String,
    args: Vec<SymExpr>,
}

impl fmt::Debug for FuncCall {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}{:?}", self.name, self.args)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
enum OpKind {
    Assign,
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    And,
    Or,
    NEq,
    Eq,
    Ge,
    GeEq,
    Lt,
    LtEq,
    Same,
    UnSame,
    Rule,
    Replace,
    Fac,
    Deriv,
}

impl OpKind {
    fn binary(value: &Wolfram) -> Option<Self> {
        use OpKind as O;
        use Wolfram as W;
        Some(match value {
            W::Assign => O::Assign,
            W::Add => O::Add,
            W::Sub => O::Sub,
            W::Mul => O::Mul,
            W::Div => O::Div,
            W::Pow => O::Pow,
            W::And => O::And,
            W::Or => O::Or,
            W::Eq => O::Eq,
            W::NEq => O::NEq,
            W::Ge => O::Ge,
            W::GeEq => O::GeEq,
            W::Lt => O::Lt,
            W::LtEq => O::LtEq,
            W::Same => O::Same,
            W::UnSame => O::UnSame,
            W::Rule => O::Rule,
            W::Replace => O::Replace,
            _ => return None,
        })
    }

    fn unary(value: &Wolfram) -> Option<Self> {
        use OpKind as O;
        use Wolfram as W;
        Some(match value {
            W::Sub => O::Sub,
            _ => return None,
        })
    }
}

#[derive(Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct List {
    args: Vec<SymExpr>,
}

impl List {
    fn for_each<F>(&self, f: F)
    where
        F: Fn(&SymExpr) + Copy,
    {
        self.args.iter().for_each(|e| e.for_each(f))
    }
}

impl fmt::Debug for List {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self.args)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
enum ScopeKind {
    With,
    Block,
    Module,
}

impl TryFrom<&str> for ScopeKind {
    type Error = ();

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        if value == "With" {
            Ok(ScopeKind::With)
        } else if value == "Block" {
            Ok(ScopeKind::Block)
        } else if value == "Module" {
            Ok(ScopeKind::Module)
        } else {
            Err(())
        }
    }
}

#[derive(Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
struct Scope {
    kind: ScopeKind,
    vars: List,
    pat: Pattern,
    //expr: Box<SymExpr>,
    //cond: Option<Box<SymExpr>>,
}

impl fmt::Debug for Scope {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}[{:?}, {:?}]", self.kind, self.vars, self.pat)
    }
}

#[derive(Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
enum SymExpr {
    Ident(String),
    IdentRestr(String, Vec<SymExpr>),
    IdentField(Vec<String>),
    Pat(String),
    PatOpt(String),
    PatRestr(String, Vec<SymExpr>),
    Integer(u64),
    Float(F64),
    Str(String),
    Slot(Option<u64>),

    BinOp(OpKind, Box<Self>, Box<Self>),
    UnryOp(OpKind, Box<Self>),

    List(List),
    Part(Box<SymExpr>, Vec<SymExpr>),
    FuncDef(FuncDef),
    FuncCall(FuncCall),
    Call(Box<SymExpr>, Vec<SymExpr>),

    Scope(Scope),
    MatchQ(Box<SymExpr>, Pattern),
    // a /. b -> c
    //ReplaceAll(Box<SymExpr>, Box<SymExpr>, Box<SymExpr>),
    Compnd(Vec<SymExpr>),
}

impl SymExpr {
    fn for_each<F>(&self, f: F)
    where
        F: Fn(&SymExpr) + Copy,
    {
        use SymExpr as S;
        f(self);
        match self {
            S::Compnd(e) | S::IdentRestr(_, e) | S::PatRestr(_, e) | S::List(List { args: e }) => {
                e.iter().for_each(|e| e.for_each(f))
            }
            S::BinOp(_, lhs, rhs) => {
                lhs.for_each(f);
                rhs.for_each(f);
            }
            S::UnryOp(_, e) => e.for_each(f),
            S::Part(e, indx) => {
                e.for_each(f);
                indx.iter().for_each(|e| e.for_each(f));
            }
            S::FuncDef(FuncDef { call, pat }) => {
                call.for_each(f);
                pat.for_each(f);
            }
            S::FuncCall(call) => call.for_each(f),
            S::Call(e, args) => {
                e.for_each(f);
                args.iter().for_each(|e| e.for_each(f));
            }
            S::Scope(Scope { kind, vars, pat }) => {
                vars.args.iter().for_each(|e| e.for_each(f));
                pat.for_each(f);
            }
            S::MatchQ(e, p) => {
                e.for_each(f);
                p.for_each(f);
            }
            _ => (),
        }
    }
}

impl fmt::Debug for SymExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use SymExpr as S;
        match self {
            S::Ident(id) => write!(f, "Id({})", id),
            S::IdentRestr(id, restr) => write!(f, "Id({id}, {restr:?})"),
            S::IdentField(ids) => write!(f, "Id({})", ids.as_slice().join("::")),
            S::Pat(pat) => write!(f, "Pat({})", pat),
            S::PatOpt(pat) => write!(f, "PatOpt({})", pat),
            S::PatRestr(pat, restr) => write!(f, "Pat({pat}, {restr:?})"),
            S::Integer(i) => write!(f, "{}", i),
            S::Float(v) => write!(f, "{}", v),
            S::Str(str) => write!(f, "{}", str),
            S::BinOp(op, lhs, rhs) => write!(f, "{op:?}[{rhs:?}, {lhs:?}]"),
            S::UnryOp(op, val) => write!(f, "{op:?}[{val:?}]"),
            S::List(l) => write!(f, "{{{l:?}}}"),
            S::Part(list, indx) => write!(f, "{:?}[[{:?}]]", list, indx),
            S::FuncDef(fd) => write!(f, "FuncDef[{:?}]", fd),
            S::FuncCall(call) => write!(f, "Call[{:?}]", call),
            S::Call(expr, args) => write!(f, "Call[{expr:?}[{args:?}]]"),
            S::Scope(scope) => write!(f, "{scope:?}"),
            S::MatchQ(expr, from) => write!(f, "MatchQ[{expr:?}, {from:?}]"),
            //S::ReplaceAll(a, b, c) => write!(f, "ReplaceAll[{a:?} /. {b:?} -> {c:?}]"),
            S::Compnd(exprs) => write!(f, "Compnd[{exprs:?}]"),
            S::Slot(None) => write!(f, "#"),
            S::Slot(Some(i)) => write!(f, "#{i}"),
        }
    }
}

impl SymExpr {
    fn binary(op: &Wolfram, lhs: Self, rhs: Self) -> Self {
        match OpKind::binary(op) {
            Some(op) => Self::BinOp(op, lhs.into(), rhs.into()),
            _ => error!("unexpected binary op: {:?}", op),
        }
    }

    fn unary(op: &Wolfram, val: Self) -> Self {
        match OpKind::unary(op) {
            Some(op) => Self::UnryOp(op, val.into()),
            _ => error!("unexpected binary op: {:?}", op),
        }
    }
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
struct WolframContext {
    exprs: Vec<SymExpr>,
}

impl WolframContext {

    fn push_expr(&mut self, e: SymExpr) {
        self.exprs.push(e);
    }
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
struct WolframFiles {
    func_defs: HashMap<String, HashSet<FuncDef>>,
    func_calls: HashMap<String, HashSet<FuncCall>>,
    files: Vec<(String, WolframContext)>,
}

impl WolframFiles {
    fn next_file(&mut self, name: String) {
        self.files.push((name.into(), Default::default()))
    }

    fn func_defs(&self) -> Vec<String> {
        self.func_defs.keys().cloned().collect()
    }

    fn register_expr(&mut self, e: &SymExpr) {
        match e {
            SymExpr::FuncDef(fd) => self.func_def(fd),
            SymExpr::FuncCall(fc) => self.func_call(fc),
            _ => (),
        }
    }

    fn push_expr(&mut self, e: SymExpr) {
        self.register_expr(&e);
        self.files.last_mut().unwrap().1.push_expr(e)
    }

    fn func_call(&mut self, fc: &FuncCall) {
        self.func_calls
            .entry(fc.name.clone())
            .and_modify(|calls| {
                calls.insert(fc.clone());
            })
            .or_insert([fc.clone()].into_iter().collect());
    }
    fn func_def(&mut self, fd: &FuncDef) {
        self.func_defs
            .entry(fd.call.name.clone())
            .and_modify(|calls| {
                calls.insert(fd.clone());
            })
            .or_insert([fd.clone()].into_iter().collect());
    }

    fn func_calls(&self) -> Vec<String> {
        self.func_calls.keys().cloned().collect()
    }

    fn builtin_func_calls(&self) -> Vec<String> {
        self.func_calls
            .keys()
            .filter(|fc| !self.func_defs.contains_key(*fc))
            .cloned()
            .collect()
    }
}

struct Parser {
    current: (Wolfram, ops::Range<usize>),
    cntxt: WolframFiles,
    tok_count: usize,
    token_iter: std::iter::Peekable<Box<dyn Iterator<Item = (Wolfram, ops::Range<usize>)>>>,
}

impl fmt::Debug for Parser {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Parser")
            .field("current", &self.current)
            .field("context", &self.cntxt)
            .finish()
    }
}

macro_rules! assert_tok {
    ($t: ident, $e:expr) => {{
        if !$t.check($e) {
            let line = $t.slice_line();
            let line_part = $t.slice(15);
            error!(
                "expected: {:?}, found: {:?}\nhere: {line_part}\nline: {line}",
                $e,
                $t.current()
            );
        } else {
            $t.advance();
        }
    }};
}

impl Parser {
    fn new(mut token_iter: impl Iterator<Item = (Wolfram, ops::Range<usize>)> + 'static) -> Self {
        let current = token_iter.next().expect("empty iterator");
        let token_iter: Box<dyn Iterator<Item = (Wolfram, ops::Range<usize>)>> =
            Box::new(token_iter);
        Self {
            current,
            cntxt: Default::default(),
            tok_count: 0,
            token_iter: token_iter.peekable(),
        }
    }

    fn next(&mut self) -> Option<&Wolfram> {
        self.token_iter.next().map(|t| {
            self.current = t;
            self.tok_count += 1;
            &self.current.0
        })
    }

    fn skip_comment(&mut self) -> Option<&Wolfram> {
        let mut n = self.current.0.clone();
        while let Wolfram::Comment(c) = n {
            if c.contains("FILE:") {
                self.cntxt.next_file(c.clone())
            }
            n = self.next()?.clone();
        }
        Some(&self.current.0)
    }

    fn skip_comment_nl(&mut self) -> Option<&Wolfram> {
        let mut n = self.current.0.clone();
        while n.skippable() {
            if let Wolfram::Comment(c) = n {
                if c.contains("FILE:") {
                    self.cntxt.next_file(c.clone())
                }
            }
            n = self.next()?.clone();
        }
        Some(&self.current.0)
    }

    fn advance(&mut self) {
        self.next();
        let _ = self.skip_comment();
    }

    fn peek(&mut self) -> Option<&Wolfram> {

        self.token_iter.peek().map(|(t, _)| t)
    }

    fn check_peek(&mut self, t: &Wolfram) -> bool {
        self.peek().is_some_and(|tok| tok == t)
    }

    fn reached_end(&mut self) -> bool {
        self.token_iter.peek().is_none()
    }

    fn check(&mut self, expected: &Wolfram) -> bool {
        if self.current() != expected {
            false
        } else {
            true
        }
    }

    fn slice_line(&self) -> &'static str {
        let src = INTEGRATION_RULES;
        let tok_span = self.current_span();
        let line_start = src[..tok_span.start].rfind('\n').map_or(0, |pos| pos + 1);
        let line_end = src[tok_span.end..]
            .find('\n')
            .map_or(src.len(), |pos| tok_span.end + pos);
        &src[line_start..line_end]
    }

    fn slice(&self, len: usize) -> &'static str {
        let src = INTEGRATION_RULES;
        let mut span = self.current.1.clone();
        span.start = std::cmp::max(0isize, span.start as isize - len as isize) as usize;
        span.end = std::cmp::min(src.len(), span.end + len);
        &src[span]
    }

    fn assert(&mut self, expected: &Wolfram) {
        if self.current() != expected {
            error!(
                "found: {:?}, expected: {:?}\nline: {}",
                self.current.0,
                expected,
                self.slice_line()
            );
        }
        self.advance();
    }

    fn current(&self) -> &Wolfram {
        &self.current.0
    }

    fn current_span(&self) -> &ops::Range<usize> {
        &self.current.1
    }
}

fn parse_func_def(call: FuncCall, t: &mut Parser) -> FuncDef {
    use Wolfram as W;
    assert_tok!(t, &W::Def);
    let pat = parse_pattern(t);
    let res = FuncDef { call, pat };
    //t.cntxt.func_def(&res);
    res
}

fn parse_scope(kind: ScopeKind, t: &mut Parser) -> Scope {
    use Wolfram as W;
    assert_tok!(t, &W::LBracket);
    let vars = parse_list(t);
    assert_tok!(t, &W::Comma);
    let pat = parse_pattern(t);
    assert_tok!(t, &W::RBracket);
    Scope { kind, vars, pat }
}

fn parse_matchq(t: &mut Parser) -> SymExpr {
    use Wolfram as W;
    assert_tok!(t, &W::LBracket);
    let expr = parse_expr(t);
    assert_tok!(t, &W::Comma);
    let pat = parse_pattern(t);
    assert_tok!(t, &W::RBracket);
    SymExpr::MatchQ(expr.into(), pat)
}

fn parse_pattern(t: &mut Parser) -> Pattern {
    use Wolfram as W;
    let pat = parse_expr(t).into();
    let mut cond = None;

    if t.check(&W::Cond) {
        assert_tok!(t, &W::Cond);
        cond = Box::new(parse_expr(t)).into();
    }
    Pattern { pat, cond }
}

fn parse_args(t: &mut Parser) -> Vec<SymExpr> {
    use Wolfram as W;
    let mut args = vec![];
    assert_tok!(t, &W::LBracket);
    loop {
        if t.check(&W::RBracket) {
            break;
        }
        let arg = parse_expr(t);
        args.push(arg);
        if t.check(&W::RBracket) {
            break;
        }
        assert_tok!(t, &W::Comma)
    }
    assert_tok!(t, &W::RBracket);
    args
}

fn parse_call(name: String, t: &mut Parser) -> SymExpr {
    use Wolfram as W;

    if let Ok(sk) = ScopeKind::try_from(name.as_str()) {
        return SymExpr::Scope(parse_scope(sk, t));
    } else if name == "MatchQ" {
        return parse_matchq(t);
    }

    let args = parse_args(t);
    let call = FuncCall { name, args };
    if t.check(&W::Def) {
        let fd = SymExpr::FuncDef(parse_func_def(call, t));
        t.cntxt.register_expr(&fd);
        fd
    } else {
        //t.cntxt.func_call(&call);
        let fc = SymExpr::FuncCall(call);
        t.cntxt.register_expr(&fc);
        fc
    }
}

fn parse_ident(str: String, t: &mut Parser) -> SymExpr {
    use Wolfram as W;
    if str.ends_with('_') {
        // is a pattern
        if t.check(&W::Dot) {
            t.advance();
            SymExpr::PatOpt(str)
        } else if t.check(&W::Colon) {
            t.advance();
            let r = parse_expr(t);
            SymExpr::PatRestr(str, vec![r])
        } else {
            SymExpr::Pat(str)
        }
    } else if t.check(&W::Colon) {
        t.advance();
        let r = parse_expr(t);
        SymExpr::IdentRestr(str, vec![r])
    } else if t.check(&W::DoublColon) {
        let mut ids = vec![str];
        loop {
            assert_tok!(t, &W::DoublColon);
            if let W::Ident(id) = t.current() {
                ids.push(id.clone());
            } else {
                error!("expected identifier, found: {:?}", t.current());
            }
            if !t.check(&W::DoublColon) {
                break;
            }
        }
        SymExpr::IdentField(ids)
    } else {
        SymExpr::Ident(str)
    }
}

fn parse_list(t: &mut Parser) -> List {
    use Wolfram as W;
    let mut args = vec![];
    assert_tok!(t, &W::LBrace);
    loop {
        if t.check(&W::RBrace) {
            break;
        }
        let arg = parse_expr(t);
        args.push(arg);
        t.skip_comment_nl();
        if t.check(&W::RBrace) {
            break;
        }
        assert_tok!(t, &W::Comma)
    }
    assert_tok!(t, &W::RBrace);
    List { args }
}

//fn parse_opt_mul(lhs: SymExpr, t: &mut Parser) -> SymExpr {
//    use Wolfram as W;
//    match t.current() {
//        W::LParen | W::Integer(_) | W::Ident(_) => {
//            let rhs = parse_operand(t);
//            SymExpr::binary(&W::Mul, lhs, rhs)
//        }
//        _ => lhs,
//    }
//}

fn parse_operand(t: &mut Parser) -> SymExpr {
    use SymExpr as SE;
    use Wolfram as W;

    match t.current().clone() {
        W::Ident(v) => {
            t.advance();
            if t.check(&W::LBracket) && !t.check_peek(&W::LBracket) {
                parse_call(v, t)
            } else {
                parse_ident(v, t)
            }
        }
        W::Str(v) => {
            t.advance();
            SE::Str(v)
        }
        W::Integer(v) => {
            t.advance();
            SE::Integer(v)
        }
        W::Float(v) => {
            t.advance();
            SE::Float(v.into())
        }
        //W::Float(v) => {
        //    t.advance();
        //    SE::Float(v)
        //}
        W::LParen => {
            t.advance();
            let expr = parse_expr(t);
            assert_tok!(t, &W::RParen);
            expr
            //let expr = parse_expr(t);
            //if !t.check(&W::RParen) {
            //    let expr = parse_opt_mul(expr, t);
            //    assert_tok!(t, &W::RParen);
            //    expr
            //} else {
            //    assert_tok!(t, &W::RParen);
            //    let expr = parse_opt_mul(expr, t);
            //    expr
            //}
        }
        W::LBrace => SE::List(parse_list(t)),
        W::Add => {
            t.advance();
            parse_expr(t)
        }
        W::Sub => {
            t.advance();
            let expr = parse_bin_expr(t, W::Mul.op_prec() + 1);
            SE::unary(&W::Sub, expr)
        }
        W::Slot => {
            t.advance();
            if let W::Integer(i) = t.current().clone() {
                t.advance();
                SE::Slot(i.into())
            } else {
                SE::Slot(None)
            }
        }
        tok => {
            if tok.skippable() {
                t.skip_comment_nl();
                parse_operand(t)
            } else {
                error!(
                    "unexpected token: {:?}\nhere: {}\nline: {}",
                    tok,
                    t.slice(15),
                    t.slice_line()
                )
            }
        }
    }
}

fn parse_unary(expr: SymExpr, t: &mut Parser) -> SymExpr {
    use Wolfram as W;
    //let expr = parse_operand(t);
    let unary = if t.check(&W::LBracket) && t.check_peek(&W::LBracket) {
        assert_tok!(t, &W::LBracket);
        //assert_tok!(t, &W::LBracket);
        let indx = parse_args(t);
        //assert_tok!(t, &W::RBracket);
        assert_tok!(t, &W::RBracket);
        SymExpr::Part(expr.into(), indx.into())
    } else if t.check(&W::LBracket) {
        let args = parse_args(t);
        SymExpr::Call(expr.into(), args)
    } else if t.check(&W::Not) {
        assert_tok!(t, &W::Not);
        SymExpr::UnryOp(OpKind::Fac, expr.into())
    } else if t.check(&W::Deriv) {
        assert_tok!(t, &W::Deriv);
        SymExpr::UnryOp(OpKind::Deriv, expr.into())
    } else {
        return expr;
    };
    parse_unary(unary, t)
}

fn parse_compl_expr(t: &mut Parser) -> SymExpr {
    let expr = parse_operand(t);
    parse_unary(expr, t)
}

fn parse_compound(t: &mut Parser) -> SymExpr {
    use Wolfram as W;
    let expr = parse_compl_expr(t);
    if !t.check(&W::Semicolon) {
        return expr;
    }

    let mut exprs = vec![expr];
    while t.check(&W::Semicolon) {
        assert_tok!(t, &W::Semicolon);
        let e = parse_compl_expr(t);
        exprs.push(e)
    }
    SymExpr::Compnd(exprs)
}

//fn get_rhs_for_mul(e: SymExpr) -> (Option<SymExpr>, SymExpr) {
//    use OpKind as O;
//    use SymExpr as S;
//    match e {
//        S::BinOp(O::Mul | O::Add | O::Sub, lhs, rhs) => (Some(lhs), rhs),
//        lhs => (None, lhs),
//    }
//}

fn parse_bin_expr(t: &mut Parser, prec_in: u32) -> SymExpr {
    use Wolfram as W;
    let mut lhs = parse_compound(t);
    loop {
        let mut op = t.current().clone();

        match &op {
            W::Ident(_) => {
                //if p.is_some_and(|p| p.is_op() || p == &W::LBracket) {
                op = W::Mul;
                if op.op_prec() < prec_in {
                    break;
                }
                //} else {
                //    break;
                //}
            }
            W::LParen => {
                op = W::Mul;
                if op.op_prec() < prec_in {
                    break;
                }
            }
            _ => {
                if op.op_prec() < prec_in {
                    break;
                }
                t.advance();
            }
        }
        let rhs = parse_bin_expr(t, op.op_prec() + 1);
        lhs = SymExpr::binary(&op, lhs, rhs);
    }

    lhs
}

fn parse_expr(t: &mut Parser) -> SymExpr {
    //t.skip_comment_nl();
    parse_bin_expr(t, 0 + 1)
}

pub fn load_rubi() {
    let lexer = Wolfram::lexer(INTEGRATION_RULES);
    let tokens = lexer.spanned().map(|(tok, span)| {
        if let Ok(tok) = tok {
            (tok, span)
        } else {
            panic!("could not lex: {:?}", INTEGRATION_RULES.slice(span));
        }
    });

    let mut t = Parser::new(tokens);
    let mut count = 0;
    loop {
        let expr = parse_expr(&mut t);
        t.cntxt.push_expr(expr);
        t.skip_comment_nl();
        if t.reached_end() {
            break;
        }
        //assert_tok!(t, &Wolfram::NL);
    }
    //println!("{:?}", t.cntxt.builtin_func_calls());
    //let f1 = t.cntxt.files.first().unwrap();
    //println!("{:?}", f1.0);
    //f1.1.exprs.iter().for_each(|e| {
    //    println!("{:?}", e);
    //})
    //t.cntxt.files.iter().for_each(|f| println!("{:?}", f.1.builtin_func_calls()));

    //let exprs_str = serde_json::to_string(&exprs).unwrap();
    //std::fs::write("exprs.json", &exprs_str).expect("Unable to write file");

    //println!("func. calls: {:?}", t.cntxt.builtin_func_calls());
    //let tokens: Vec<_> = tokens.collect();
    //println!("{:?}", &tokens[0..10]);
}
