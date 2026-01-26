use ahash::AHasher;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use fxhash::hash64 as fxhash64;
use rand::prelude::*;
use seqhash::SeqHash;
use std::collections::HashMap;
use std::hash::Hasher;
use std::hint::black_box;

const BASES: [u8; 5] = [b'A', b'C', b'G', b'T', b'N'];

fn generate_random_parents(num_parents: usize, seq_len: usize) -> Vec<Vec<u8>> {
    let mut rng = rand::rng();
    (0..num_parents)
        .map(|_| {
            (0..seq_len)
                .map(|_| BASES[rng.random_range(0..5)])
                .collect()
        })
        .collect()
}

fn generate_queries(parents: &[Vec<u8>], num_queries: usize, exact_ratio: f64) -> Vec<Vec<u8>> {
    let mut rng = rand::rng();
    let mut queries = Vec::with_capacity(num_queries);
    let seq_len = parents[0].len();

    for _ in 0..num_queries {
        let parent = &parents[rng.random_range(0..parents.len())];
        let mut query = parent.clone();

        if rng.random::<f64>() >= exact_ratio {
            // Introduce single mutation
            let pos = rng.random_range(0..seq_len);
            let original = query[pos];
            let mut new_base = BASES[rng.random_range(0..5)];
            while new_base == original {
                new_base = BASES[rng.random_range(0..5)];
            }
            query[pos] = new_base;
        }

        queries.push(query);
    }

    queries
}

/// Naive implementation using HashMap<Vec<u8>, usize> for comparison.
/// Stores all parent sequences and all single-mismatch variants explicitly.
struct NaiveSeqHash {
    /// Maps sequence -> parent index (usize::MAX means ambiguous)
    lookup: HashMap<Vec<u8>, usize>,
}

impl NaiveSeqHash {
    fn new(parents: &[Vec<u8>]) -> Self {
        let seq_len = parents[0].len();
        // Estimate: parents + 4*seq_len mutations per parent
        let estimated_entries = parents.len() * (1 + 4 * seq_len);
        let mut lookup: HashMap<Vec<u8>, usize> = HashMap::with_capacity(estimated_entries);

        // Insert all parents
        for (idx, parent) in parents.iter().enumerate() {
            if let Some(&existing) = lookup.get(parent) {
                if existing != usize::MAX {
                    // Mark as ambiguous
                    lookup.insert(parent.clone(), usize::MAX);
                }
            } else {
                lookup.insert(parent.clone(), idx);
            }
        }

        // Insert all single-mismatch variants
        for (parent_idx, parent) in parents.iter().enumerate() {
            for pos in 0..seq_len {
                let original_base = parent[pos];
                for &new_base in &BASES {
                    if new_base == original_base {
                        continue;
                    }

                    let mut mutant = parent.clone();
                    mutant[pos] = new_base;

                    if let Some(&existing) = lookup.get(&mutant) {
                        if existing != usize::MAX && existing != parent_idx {
                            // Collision with different parent - mark ambiguous
                            lookup.insert(mutant, usize::MAX);
                        }
                    } else {
                        lookup.insert(mutant, parent_idx);
                    }
                }
            }
        }

        NaiveSeqHash { lookup }
    }

    fn query(&self, seq: &[u8]) -> Option<usize> {
        match self.lookup.get(seq) {
            Some(&idx) if idx != usize::MAX => Some(idx),
            _ => None,
        }
    }
}

// ============================================================================
// Construction benchmarks
// ============================================================================

fn bench_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("construction");
    group.sample_size(10);

    let parents = generate_random_parents(50_000, 49);

    group.bench_function("seqhash", |b| {
        b.iter(|| SeqHash::new(black_box(&parents)).unwrap())
    });

    group.bench_function("naive_hashmap", |b| {
        b.iter(|| NaiveSeqHash::new(black_box(&parents)))
    });

    group.finish();
}

// ============================================================================
// Query benchmarks (batch)
// ============================================================================

fn bench_query_batch(c: &mut Criterion) {
    let mut group = c.benchmark_group("query_batch_100k");

    let parents = generate_random_parents(50_000, 49);
    let queries = generate_queries(&parents, 100_000, 0.90);

    let seqhash_index = SeqHash::new(&parents).unwrap();
    let naive_index = NaiveSeqHash::new(&parents);

    group.bench_function("seqhash", |b| {
        b.iter(|| {
            for q in &queries {
                black_box(seqhash_index.query(q));
            }
        })
    });

    group.bench_function("naive_hashmap", |b| {
        b.iter(|| {
            for q in &queries {
                black_box(naive_index.query(q));
            }
        })
    });

    group.finish();
}

// ============================================================================
// Query benchmarks (single)
// ============================================================================

fn bench_query_single(c: &mut Criterion) {
    let mut group = c.benchmark_group("query_single");

    let parents = generate_random_parents(50_000, 49);
    let queries = generate_queries(&parents, 1000, 0.90);

    let seqhash_index = SeqHash::new(&parents).unwrap();
    let naive_index = NaiveSeqHash::new(&parents);

    group.bench_function("seqhash", |b| {
        let mut i = 0;
        b.iter(|| {
            let q = &queries[i % queries.len()];
            i += 1;
            black_box(seqhash_index.query(q))
        })
    });

    group.bench_function("naive_hashmap", |b| {
        let mut i = 0;
        b.iter(|| {
            let q = &queries[i % queries.len()];
            i += 1;
            black_box(naive_index.query(q))
        })
    });

    group.finish();
}

// ============================================================================
// Scaling benchmarks - vary number of parents
// ============================================================================

fn bench_construction_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("construction_scaling");
    group.sample_size(10); // Reduce sample size for slower benchmarks

    for num_parents in [1_000, 10_000, 50_000] {
        let parents = generate_random_parents(num_parents, 49);

        group.bench_with_input(
            BenchmarkId::new("seqhash", num_parents),
            &parents,
            |b, parents| b.iter(|| SeqHash::new(black_box(parents)).unwrap()),
        );

        group.bench_with_input(
            BenchmarkId::new("naive_hashmap", num_parents),
            &parents,
            |b, parents| b.iter(|| NaiveSeqHash::new(black_box(parents))),
        );
    }

    group.finish();
}

// ============================================================================
// Hash function comparison
// ============================================================================

fn hash_fnv1a(seq: &[u8]) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    for &b in seq {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

fn hash_wyhash(seq: &[u8]) -> u64 {
    wyhash::wyhash(seq, 0)
}

fn hash_xxh3(seq: &[u8]) -> u64 {
    xxhash_rust::xxh3::xxh3_64(seq)
}

fn hash_ahash(seq: &[u8]) -> u64 {
    let mut hasher = AHasher::default();
    hasher.write(seq);
    hasher.finish()
}

fn hash_fxhash(seq: &[u8]) -> u64 {
    fxhash64(seq)
}

fn bench_hash_functions(c: &mut Criterion) {
    let mut group = c.benchmark_group("hash_functions");

    // Generate test sequences (49bp like our use case)
    let sequences: Vec<Vec<u8>> = (0..10000)
        .map(|i| {
            let mut rng = rand::rngs::StdRng::seed_from_u64(i);
            (0..49).map(|_| BASES[rng.random_range(0..4)]).collect()
        })
        .collect();

    group.bench_function("fnv1a", |b| {
        b.iter(|| {
            for seq in &sequences {
                black_box(hash_fnv1a(seq));
            }
        })
    });

    group.bench_function("wyhash", |b| {
        b.iter(|| {
            for seq in &sequences {
                black_box(hash_wyhash(seq));
            }
        })
    });

    group.bench_function("xxh3", |b| {
        b.iter(|| {
            for seq in &sequences {
                black_box(hash_xxh3(seq));
            }
        })
    });

    group.bench_function("ahash", |b| {
        b.iter(|| {
            for seq in &sequences {
                black_box(hash_ahash(seq));
            }
        })
    });

    group.bench_function("fxhash", |b| {
        b.iter(|| {
            for seq in &sequences {
                black_box(hash_fxhash(seq));
            }
        })
    });

    group.finish();
}

// ============================================================================
// Memory usage estimation (not a benchmark, but useful info)
// ============================================================================

fn bench_memory_report(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_estimate");
    group.sample_size(10);

    let parents = generate_random_parents(50_000, 49);

    // Build both indices to report their sizes
    let seqhash_index = SeqHash::new(&parents).unwrap();
    let naive_index = NaiveSeqHash::new(&parents);

    println!("\n=== Memory Usage Estimates (50k parents, 49bp) ===");
    println!("SeqHash entries: {}", seqhash_index.num_entries());
    println!("SeqHash ambiguous: {}", seqhash_index.num_ambiguous());
    println!("Naive HashMap entries: {}", naive_index.lookup.len());

    // Rough memory estimates
    // SeqHash: parents (50k * 49) + lookup (entries * ~16 bytes for hashmap overhead + 8 byte entry)
    let seqhash_parent_mem = 50_000 * 49;
    let seqhash_lookup_mem = seqhash_index.num_entries() * 24; // ~24 bytes per hashmap entry
    let seqhash_total = seqhash_parent_mem + seqhash_lookup_mem;

    // Naive: lookup (entries * (49 + 8 + ~16 bytes overhead))
    let naive_entry_size = 49 + 8 + 24; // Vec<u8> data + usize + HashMap overhead
    let naive_total = naive_index.lookup.len() * naive_entry_size;

    println!(
        "\nSeqHash estimated memory: {} MB",
        seqhash_total / 1_000_000
    );
    println!("Naive estimated memory: {} MB", naive_total / 1_000_000);
    println!(
        "Memory ratio (naive/seqhash): {:.1}x",
        naive_total as f64 / seqhash_total as f64
    );
    println!();

    // Dummy benchmark just to make criterion happy
    group.bench_function("noop", |b| b.iter(|| black_box(1 + 1)));
    group.finish();
}

criterion_group!(
    benches,
    bench_construction,
    bench_query_batch,
    bench_query_single,
    bench_construction_scaling,
    bench_hash_functions,
    bench_memory_report,
);
criterion_main!(benches);
