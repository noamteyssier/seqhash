# seqhash

[![MIT licensed](https://img.shields.io/badge/license-MIT-blue.svg)](./LICENSE.md)
[![Crates.io](https://img.shields.io/crates/d/seqhash?color=orange&label=crates.io)](https://crates.io/crates/seqhash)
[![docs.rs](https://img.shields.io/docsrs/seqhash?color=green&label=docs.rs)](https://docs.rs/seqhash/latest/seqhash/)

Fast mismatch-tolerant sequence lookup with disambiguation.

`seqhash` is a high-performance Rust library for building indices that support approximate matching of DNA sequences.
Given a set of parent sequences, it constructs an index that can determine whether a query sequence matches any parent exactly *or* differs by exactly one substitution.
It also detects and rejects ambiguous cases where a query could map to multiple parents.

This builds on the work of my original library [`disambiseq`](https://github.com/noamteyssier/disambiseq) but is significantly more memory efficient, faster, and easier to use (in my opinion).

## The Problem

In many bioinformatics applications (barcode demultiplexing, guide RNA mapping, subsequence matching), you need to match observed sequences against a reference set while tolerating sequencing errors.
A common requirement is to allow up to one mismatch, but only when the match is unambiguous.

### Naive Approach

The straightforward solution is to pre-compute all possible single-base mutations for each parent sequence and store them in a hash table:

```
Parent: ACGT
Mutations: CCGT, GCGT, TCGT, NCGT,  (position 0)
           AAGT, AGGT, ATGT, ANGT,  (position 1)
           ACAT, ACCT, ACTT, ACNT,  (position 2)
           ACGA, ACGC, ACGG, ACGN   (position 3)
```

This works, but has drawbacks:

1. **Memory overhead**: Each entry stores a full copy of the sequence plus hash table overhead. With 50,000 parents, 49bp sequences, and ~200 mutations each, you're storing ~10 million sequences.

2. **Construction time**: Allocating and hashing millions of sequence copies is expensive.

3. **Query overhead**: Looking up variable-length keys requires hashing the full sequence and comparing bytes on collision.

### Alternative Approaches

#### Store parents only, generate mutations in-flight

An alternative approach is to only store a hash table of the parent sequences and for each sequence that does not match to generate their mutations on the fly.
This can reduce memory overhead at construction time, but the query time will be slower as it will require generating the mutations on the fly.

This can be minimal for small datasets, or when the positive lookup rate is high, but even for mapping rates in the 90% on normal biological datasets, the overhead of generating mutations at each base can be quite significant.

Unfortunately to handle ambiguous mismatches (i.e. a mutated sequence which has multiple possible parents), this approach requires generating *all* possible mutations every time and not just up to the first matching mutation.

#### Store parents only, calculate hamming distance

Another alternative approach is to store a hash table of the parent sequences and for each sequence that does not match to calculate the hamming distance to an existing parent sequence.

This can easily become impractical as it requires performing a full hamming distance calculation for each sequence *for each parent* and suffers from the same issues as the previous approach with respect to ambiguous mismatches.

### The seqhash Approach

`seqhash` uses a fundamentally different strategy:

1. **Store parents once**: Parent sequences are stored contiguously in a single `Vec<u8>`, indexed by position.

2. **Compact entry encoding**: Instead of storing mutation sequences, we store *metadata about the mutation* in a 64-bit integer:
   - Parent index (32 bits)
   - Mutation position (14 bits)
   - Original base (8 bits)
   - Mutated base (8 bits)
   - Flags for ambiguity/exact-match (2 bits)

3. **Hash the mutation, store the delta**: We hash mutation sequences during construction but only store the compact entry. During query, we verify matches by checking that the query differs from the parent at exactly the expected position with the expected bases.

This design provides:

- **~3× lower memory usage**: Entries are 8 bytes instead of ~80 bytes (sequence + overhead)
- **~4× faster construction**: No allocation per mutation, just compute hash and insert entry
- **~1.5× faster queries**: Fixed-size entries, better cache locality

## Usage

Add to your `Cargo.toml`:

```toml
[dependencies]
seqhash = "0.1"
```

### Basic Example

```rust
use seqhash::{SeqHash, Match};

let parents: Vec<&[u8]> = vec![
    b"ACGTACGTACGT",
    b"GGGGCCCCAAAA",
    b"TTTTAAAACCCC",
];

let index = SeqHash::new(&parents).unwrap();

// Exact match
assert!(matches!(
    index.query(b"ACGTACGTACGT"),
    Some(Match::Exact { parent_idx: 0 })
));

// Single-base mismatch (T→A at position 11)
assert!(matches!(
    index.query(b"ACGTACGTACGA"),
    Some(Match::Mismatch { parent_idx: 0, pos: 11 })
));

// No match (too many differences)
assert!(index.query(b"GGGGGGGGGGGG").is_none());
```

### Match Results

The `Match` enum tells you how a query matched:

```rust
match index.query(sequence) {
    Some(Match::Exact { parent_idx }) => {
        println!("Exact match to parent {}", parent_idx);
    }
    Some(Match::Mismatch { parent_idx, pos }) => {
        println!("One mismatch from parent {} at position {}", parent_idx, pos);
    }
    None => {
        // No match, or ambiguous (maps to multiple parents)
    }
}
```

### Handling Ambiguity

When a sequence could match multiple parents (e.g., it's one mutation away from two different parents at the same position), `seqhash` returns `None` to avoid incorrect assignments:

```rust
// Two parents that differ only at position 0
let parents: Vec<&[u8]> = vec![b"ACGTACGT", b"TCGTACGT"];
let index = SeqHash::new(&parents).unwrap();

// CCGTACGT is one mutation from both parents → ambiguous
assert!(index.query(b"CCGTACGT").is_none());
assert!(index.is_ambiguous(b"CCGTACGT"));
```

### Positional Lookup Methods

For real-world applications where target sequences appear at known or variable positions within longer reads, `seqhash` provides convenience methods:

#### Direct Indexing (`query_at`)

When the target is at a fixed, known position:

```rust
const POS: usize = 23;

for record in records {
    if let Some(m) = index.query_at(record.seq(), POS) {
        counts[m.parent_idx()] += 1;
    }
}
```

#### Positional Remapping (`query_at_with_remap`)

When the target position may drift slightly due to small indels, search in an alternating pattern around the expected position (`pos`, `pos+1`, `pos-1`, `pos+2`, `pos-2`, ...):

> *Note*: This method is useful when the small indels are expected to occur *outside* the target sequence. 

```rust
const POS: usize = 23;
const WINDOW: usize = 3;

for record in records {
    if let Some(m) = index.query_at_with_remap(record.seq(), POS, WINDOW) {
        counts[m.parent_idx()] += 1;
    }
}
```

To track position drift statistics, use `query_at_with_remap_offset` which also returns the offset where the match was found:

```rust
if let Some((m, offset)) = index.query_at_with_remap_offset(record.seq(), POS, WINDOW) {
    counts[m.parent_idx()] += 1;
    if offset != 0 {
        drift_stats[offset] += 1;
    }
}
```

#### Sliding Window (`query_sliding`)

When the target position is unknown, scan through the read looking for the first match:

```rust
for record in records {
    if let Some((m, pos)) = index.query_sliding(record.seq()) {
        counts[m.parent_idx()] += 1;
    }
}
```

To find all matches in a sequence (e.g., when multiple targets may be present), use `query_sliding_iter`:

```rust
for record in records {
    for (m, pos) in index.query_sliding_iter(record.seq()) {
        counts[m.parent_idx()] += 1;
    }
}
```

The iterator is lazy, so you can efficiently take only the first N matches with `.take(n)`.

### Builder Configuration

Use `SeqHashBuilder` for more control:

```rust
use seqhash::SeqHashBuilder;

// Exact matching only (no mismatch tolerance)
let index = SeqHashBuilder::default()
    .exact()
    .build(&parents)
    .unwrap();

// Reject sequences containing N bases
let index = SeqHashBuilder::default()
    .exclude_n()
    .build(&parents)
    .unwrap();
```

### Serialization

The `serde` feature enables saving and loading pre-built indices to disk. This is useful when you want to build an index once and reuse it across multiple runs.

Enable the feature in your `Cargo.toml`:

```toml
[dependencies]
seqhash = { version = "0.1", features = ["serde"] }
```

Then use the `save` and `load` methods:

```rust
use seqhash::SeqHash;

// Build and save an index
let parents: Vec<&[u8]> = vec![b"ACGTACGT", b"GGGGCCCC"];
let index = SeqHash::new(&parents).unwrap();
index.save("my_index.seqhash").unwrap();

// Later, load the index directly
let index = SeqHash::load("my_index.seqhash").unwrap();
let result = index.query(b"ACGTACGT");
```

The index is saved in bincode format. The recommended file extension is `.seqhash`.

With the `serde` feature enabled, you can also use any serde-compatible format (JSON,MessagePack, etc.) by serializing the `SeqHash` struct directly.

## Performance Characteristics

> *Note*: Benchmarking done on MacBook M3 Pro

Benchmarked with 50,000 parents of 49bp each:

| Metric | seqhash | Naive HashMap | Improvement |
|--------|---------|---------------|-------------|
| Construction | 270ms | 1050ms | **3.9×** |
| Query (single) | 19ns | 29ns | **1.5×** |
| Query (100k batch) | 2.4ms | 4.4ms | **1.8×** |
| Memory | 238 MB | 797 MB | **3.3×** |

## Limitations

- Maximum sequence length: 16,383 bases (14-bit position encoding)
- All parent sequences must have the same length
- Only single-base substitutions are tolerated (no indels)
- Parents must be unique

## License

MIT
