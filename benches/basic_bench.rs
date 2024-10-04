use calcu_rs::{expr as e, Expr};
use criterion::{criterion_group, criterion_main, Criterion, black_box};

fn add_ones(n: i64) {
    let mut sum = e!(0);
    for _ in 0..n {
        sum += e!(1);
    }
}

pub fn criterion_benchmark(c: &mut Criterion) {
    c.bench_function("sum 1e4", |b| b.iter(|| add_ones(black_box(1_000_000))));
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
