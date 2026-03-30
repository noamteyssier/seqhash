//! Demonstration of SplitSeqHash for high-tolerance barcode matching.
//!
//! This example shows how to use `SplitSeqHash` to match sequences with up to 3 mismatches
//! using the split-map strategy. This is useful for applications like barcode demultiplexing
//! where you need to tolerate multiple sequencing errors.

use seqhash::SplitSeqHash;

/// Maximum allowed hamming distance for matching.
const MAX_HDIST: usize = 3;

/// Maps a query sequence to a parent index if within tolerance.
///
/// This implements the "cyto pattern" - a common strategy in bioinformatics:
/// 1. Try to match both halves using fast SeqHash lookups (≤1 mismatch per half)
/// 2. If both halves agree, return the match immediately
/// 3. If they conflict, reject the match
/// 4. If only one half matches, validate the other half with hamming distance
fn map_sequence(index: &SplitSeqHash, sequence: &[u8]) -> Option<usize> {
    let result = index.query(sequence);

    // Fast path: both halves agree on the same parent
    // This handles 0-2 total mismatches (≤1 per half)
    if let Some(idx) = result.agreed_idx() {
        println!("  ✓ Fast match: both halves agree on parent {}", idx);
        return Some(idx);
    }

    // Conflict: both matched but to different parents - reject
    if result.is_conflicted() {
        println!("  ✗ Conflict: halves matched different parents");
        return None;
    }

    // Fallback: one half matched, validate the other half
    // This can handle up to 3 total mismatches
    if let Some((idx, matched_half)) = result.single_match() {
        let remaining = result.remaining_hdist(MAX_HDIST).unwrap_or(0);

        println!(
            "  → Fallback: {:?} half matched parent {}, checking {:?} half with budget {}",
            matched_half,
            idx,
            matched_half.other(),
            remaining
        );

        if let Some(true_idx) = index.is_within_hdist(sequence, idx, matched_half.other(), remaining) {
            println!("  ✓ Fallback match: parent {}", true_idx);
            return Some(true_idx);
        } else {
            println!("  ✗ Fallback failed: other half exceeds budget");
        }
    }

    println!("  ✗ No match: neither half matched");
    None
}

fn main() {
    println!("=== SplitSeqHash Demo ===\n");

    // Example barcode library (16bp each)
    let barcodes: Vec<&[u8]> = vec![
        b"ACGTACGTACGTACGT", // Barcode 0
        b"GGGGCCCCGGGGCCCC", // Barcode 1
        b"TTTTAAAATTTTAAAA", // Barcode 2
        b"CCCCGGGGCCCCGGGG", // Barcode 3
    ];

    println!("Building index with {} barcodes:", barcodes.len());
    for (i, bc) in barcodes.iter().enumerate() {
        println!("  {}: {}", i, String::from_utf8_lossy(bc));
    }

    let index = SplitSeqHash::new(&barcodes).expect("Failed to build index");

    println!(
        "\nIndex details: {} bases, split at position {}",
        index.seq_len(),
        index.split_pos()
    );
    println!("  Left half:  {} bases", index.left_len());
    println!("  Right half: {} bases\n", index.right_len());

    // Test cases demonstrating different scenarios
    let test_cases = vec![
        ("Exact match", b"ACGTACGTACGTACGT"),
        ("1 mismatch (left)", b"NCGTACGTACGTACGT"),
        ("1 mismatch (right)", b"ACGTACGTACGTACGN"),
        ("2 mismatches (both halves)", b"NCGTACGTACGTACGN"),
        ("2 mismatches (one half)", b"NNGTACGTACGTACGT"),
        ("3 mismatches (1 left, 2 right)", b"NCGTACGTACGTACNN"),
        ("4 mismatches (exceeds budget)", b"NNGTACGTACGTACNN"),
        ("No match", b"NNNNNNNNNNNNNNNN"),
    ];

    println!(
        "=== Test Cases (max hamming distance = {}) ===\n",
        MAX_HDIST
    );

    for (description, query) in test_cases {
        println!(
            "Query: {} ({})",
            String::from_utf8_lossy(query),
            description
        );

        let result = map_sequence(&index, query);

        if let Some(idx) = result {
            println!(
                "  → Matched to barcode {}: {}\n",
                idx,
                String::from_utf8_lossy(barcodes[idx])
            );
        } else {
            println!("  → No match found\n");
        }
    }

    // Demonstrate conflict detection
    println!("=== Conflict Detection ===\n");

    // Create a chimeric sequence: left half from barcode 0, right half from barcode 1
    let chimera = b"ACGTACGTGGGGCCCC";
    println!(
        "Query: {} (left from BC0, right from BC1)",
        String::from_utf8_lossy(chimera)
    );

    let result = index.query(chimera);
    println!("  Left match:  {:?}", result.left);
    println!("  Right match: {:?}", result.right);

    if result.is_conflicted() {
        println!("  → Detected conflict: halves matched different parents");
    }

    println!("\n=== Performance Characteristics ===");
    println!("- Query with 0-2 mismatches: O(1) hash lookups (fast path)");
    println!("- Query with 3 mismatches: O(1) + O(n/2) hamming scan (fallback)");
    println!("- Construction: O(n * seq_len) where n = number of barcodes");
    println!("- Memory: ~2x SeqHash storage (one per half)");
}
