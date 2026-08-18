//! Multi-length sequence matching.
//!
//! This module provides [`MultiLenSeqHash`], which manages multiple [`crate::SeqHash`]
//! indices for sequences of different lengths. This is useful when you have a dataset
//! with sequences of varying lengths (e.g., guides of 19, 20, and 21 bp) and want to
//! efficiently look up sequences by automatically routing them to the appropriate
//! length-specific hash set.
//!
//! # Use Cases
//!
//! This is designed for datasets with a small number of distinct sequence lengths
//! (typically 2-4). Common scenarios include:
//!
//! - CRISPR guide libraries with varying spacer lengths (e.g., 19-21 bp)
//! - Barcode sets with different lengths
//! - Primer libraries with length variants
//!
//! # Query Behavior
//!
//! The two main query patterns have different behaviors:
//!
//! - **`query(seq)`**: Routes to the index matching `seq.len()`. Returns `None` if no
//!   sequences of that length exist in the index.
//!
//! - **`query_at(seq, pos)`**: Checks all length groups (shortest first) and returns
//!   the first match found. This is useful when searching for embedded sequences in
//!   longer reads where the target length is unknown.
//!
//! # Global Index Preservation
//!
//! Parent indices returned by all query methods correspond to the original input order,
//! regardless of internal grouping by length. This means you can use the returned
//! `parent_idx` directly to index into your original data structures.
//!
//! # Example
//!
//! ```
//! use seqhash::MultiLenSeqHash;
//!
//! // Mixed-length guide library
//! let guides: Vec<&[u8]> = vec![
//!     b"ACGTACGTACGTACGTACGT",   // 20 bp, index 0
//!     b"GGGGCCCCGGGGCCCC",       // 16 bp, index 1
//!     b"TTTTAAAATTTTAAAATT",     // 18 bp, index 2
//!     b"ACGTACGTACGTACGT",       // 16 bp, index 3
//! ];
//!
//! let index = MultiLenSeqHash::new(&guides).unwrap();
//!
//! // Query routes to correct length group
//! let result = index.query(b"GGGGCCCCGGGGCCCC").unwrap();
//! assert_eq!(result.parent_idx(), 1);  // Original index preserved
//!
//! // query_at searches all lengths at a position
//! let read = b"NNNACGTACGTACGTACGTNNN";
//! if let Some(result) = index.query_at(read, 3) {
//!     println!("Found guide {} ({}bp) at position 3",
//!              result.parent_idx(), result.seq_len());
//! }
//! ```

use crate::{Match, SeqHash, SeqHashBuilder, SeqHashError};

/// A match result that includes the sequence length.
///
/// The `parent_idx` returned is always the global index from the original input,
/// not an internal index within a length group.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct MultiLenMatch {
    /// The global parent index (from original input order).
    parent_idx: usize,
    /// The mismatch position, if any.
    mismatch_pos: Option<usize>,
    /// The sequence length that matched.
    seq_len: usize,
}

impl MultiLenMatch {
    /// Returns the parent index (global, from original input order).
    #[inline]
    #[must_use]
    pub fn parent_idx(&self) -> usize {
        self.parent_idx
    }

    /// Returns true if this was an exact match.
    #[inline]
    #[must_use]
    pub fn is_exact(&self) -> bool {
        self.mismatch_pos.is_none()
    }

    /// Returns the mismatch position, if any.
    #[inline]
    #[must_use]
    pub fn mismatch_pos(&self) -> Option<usize> {
        self.mismatch_pos
    }

    /// Returns the hamming distance contribution of this match.
    #[inline]
    #[must_use]
    pub fn hdist(&self) -> usize {
        if self.mismatch_pos.is_some() {
            1
        } else {
            0
        }
    }

    /// Returns the sequence length that matched.
    #[inline]
    #[must_use]
    pub fn seq_len(&self) -> usize {
        self.seq_len
    }
}

/// A multi-length sequence index that manages separate [`SeqHash`] indices for each unique length.
///
/// `MultiLenSeqHash` wraps multiple [`SeqHash`] instances, one for each distinct sequence
/// length in the input. Internally, sequences are grouped by length and each group gets
/// its own optimized hash index. This allows efficient lookups while supporting mixed-length
/// datasets.
///
/// # When to Use
///
/// Use `MultiLenSeqHash` when:
/// - Your sequences have 2-4 different lengths (e.g., guides of 19, 20, 21 bp)
/// - You need mismatch-tolerant matching across all lengths
/// - You want to search for embedded sequences without knowing the target length
///
/// For single-length datasets, use [`SeqHash`] directly for simpler API and slightly
/// less overhead.
///
/// # Query Methods
///
/// | Method | Behavior |
/// |--------|----------|
/// | [`query`](Self::query) | Routes to index matching query length |
/// | [`query_at`](Self::query_at) | Checks all lengths at position, returns first match |
/// | [`query_at_with_remap`](Self::query_at_with_remap) | Like `query_at` with position drift tolerance |
/// | [`query_sliding`](Self::query_sliding) | Scans sequence for first match at any position |
///
/// # Index Preservation
///
/// All returned `parent_idx` values correspond to the original input order, not internal
/// groupings. You can safely use them to index into parallel data structures.
///
/// # Example
///
/// ```
/// use seqhash::MultiLenSeqHash;
///
/// // Sequences of different lengths
/// let parents: Vec<&[u8]> = vec![
///     b"ACGTACGT",       // 8 bp, index 0
///     b"GGGGCCCC",       // 8 bp, index 1
///     b"ACGTACGTACGT",   // 12 bp, index 2
///     b"TTTTAAAACCCC",   // 12 bp, index 3
/// ];
///
/// let index = MultiLenSeqHash::new(&parents).unwrap();
///
/// // Query routes to the correct length, returns global index
/// let result = index.query(b"ACGTACGTACGT").unwrap();
/// assert_eq!(result.parent_idx(), 2);  // Global index, not 0
///
/// // query_at checks all lengths (shortest first)
/// let read = b"NNACGTACGTNN";
/// let result = index.query_at(read, 2).unwrap();
/// assert_eq!(result.seq_len(), 8);  // Found 8bp match first
/// ```
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Default)]
pub struct MultiLenSeqHash {
    /// SeqHash indices sorted by seq_len (ascending).
    indices: Vec<SeqHash>,
    /// Maps (length_group_index, local_parent_idx) -> global_parent_idx.
    /// `global_indices[i][j]` is the global index for local index `j` in `indices[i]`.
    global_indices: Vec<Vec<usize>>,
}

/// Builder for constructing a [`MultiLenSeqHash`] index with custom configuration.
///
/// Provides the same configuration options as [`SeqHashBuilder`](crate::SeqHashBuilder),
/// applied uniformly to all length groups.
///
/// # Configuration Options
///
/// | Method | Default | Description |
/// |--------|---------|-------------|
/// | [`exact()`](Self::exact) | `false` | Only index exact matches (no 1-mismatch tolerance) |
/// | [`exclude_n()`](Self::exclude_n) | `false` | Reject sequences containing N bases |
/// | [`keep_case()`](Self::keep_case) | `false` | Preserve lowercase (default normalizes to uppercase) |
///
/// # Example
///
/// ```
/// use seqhash::MultiLenSeqHashBuilder;
///
/// let parents: Vec<&[u8]> = vec![b"ACGTACGT", b"GGGGCCCCGGGG"];
///
/// // Build with exact match only (no mismatch tolerance)
/// let index = MultiLenSeqHashBuilder::default()
///     .exact()
///     .build(&parents)
///     .unwrap();
///
/// // Exact matches work
/// assert!(index.query(b"ACGTACGT").is_some());
///
/// // But 1-mismatch queries don't
/// assert!(index.query(b"NCGTACGT").is_none());
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct MultiLenSeqHashBuilder {
    /// If true, only index exact matches (no mismatch entries).
    exact_only: bool,
    /// If true, allow N bases in sequences.
    allow_n: bool,
    /// If true, convert sequences to uppercase before indexing.
    normalize_case: bool,
}

impl MultiLenSeqHashBuilder {
    /// Configure for exact matching only (no mismatch tolerance).
    #[must_use]
    pub fn exact(mut self) -> Self {
        self.exact_only = true;
        self
    }

    /// Reject N bases in sequences.
    #[must_use]
    pub fn exclude_n(mut self) -> Self {
        self.allow_n = false;
        self
    }

    /// Preserve the case of input sequences.
    #[must_use]
    pub fn keep_case(mut self) -> Self {
        self.normalize_case = false;
        self
    }

    /// Build the [`MultiLenSeqHash`] index from the given parent sequences.
    ///
    /// Sequences are grouped by length, and a separate [`SeqHash`] is created for each group.
    /// Parent indices in query results correspond to the original input order.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - No parent sequences are provided
    /// - Any sequence length exceeds 16383
    /// - Duplicate parent sequences exist within a length group
    /// - Sequences contain invalid bases
    pub fn build<S: AsRef<[u8]>>(self, parents: &[S]) -> Result<MultiLenSeqHash, SeqHashError> {
        MultiLenSeqHash::build_internal(parents, self.exact_only, self.allow_n, self.normalize_case)
    }
}

impl MultiLenSeqHash {
    /// Construct a new index from parent sequences of varying lengths.
    ///
    /// Sequences are grouped by length, and a separate [`SeqHash`] is created for each group.
    /// Parent indices in query results correspond to the original input order.
    ///
    /// This uses default settings (allows 1 mismatch, allows N bases, normalizes case).
    ///
    /// For more control, use [`MultiLenSeqHashBuilder`].
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - No parent sequences are provided
    /// - Any sequence length exceeds 16383
    /// - Duplicate parent sequences exist within a length group
    /// - Sequences contain invalid bases
    pub fn new<S: AsRef<[u8]>>(parents: &[S]) -> Result<Self, SeqHashError> {
        Self::build_internal(parents, false, true, true)
    }

    /// Internal build function.
    fn build_internal<S: AsRef<[u8]>>(
        parents: &[S],
        exact_only: bool,
        allow_n: bool,
        normalize_case: bool,
    ) -> Result<Self, SeqHashError> {
        if parents.is_empty() {
            return Err(SeqHashError::EmptyParents);
        }

        // Group sequences by length, storing (sequence, global_index)
        let mut by_length: std::collections::BTreeMap<usize, Vec<(&[u8], usize)>> =
            std::collections::BTreeMap::new();

        for (global_idx, parent) in parents.iter().enumerate() {
            let seq = parent.as_ref();
            by_length
                .entry(seq.len())
                .or_default()
                .push((seq, global_idx));
        }

        // Build a SeqHash for each length group (BTreeMap iterates in sorted order)
        let mut indices = Vec::with_capacity(by_length.len());
        let mut global_indices = Vec::with_capacity(by_length.len());

        for (_len, seqs_with_indices) in by_length {
            // Extract just the sequences for SeqHash construction
            let seqs: Vec<&[u8]> = seqs_with_indices.iter().map(|(seq, _)| *seq).collect();

            // Extract global indices in the same order
            let globals: Vec<usize> = seqs_with_indices.iter().map(|(_, idx)| *idx).collect();

            let mut builder = SeqHashBuilder::default();
            if exact_only {
                builder = builder.exact();
            }
            if !allow_n {
                builder = builder.exclude_n();
            }
            if !normalize_case {
                builder = builder.keep_case();
            }

            let index = builder.build(&seqs)?;
            indices.push(index);
            global_indices.push(globals);
        }

        Ok(Self {
            indices,
            global_indices,
        })
    }

    /// Convert a local match to a MultiLenMatch with global parent index.
    #[inline]
    fn to_global_match(&self, inner: Match, len_group_idx: usize, seq_len: usize) -> MultiLenMatch {
        let local_idx = inner.parent_idx();
        let global_idx = self.global_indices[len_group_idx][local_idx];

        MultiLenMatch {
            parent_idx: global_idx,
            mismatch_pos: inner.mismatch_pos(),
            seq_len,
        }
    }

    /// Query a sequence by routing to the appropriate length-specific index.
    ///
    /// Returns `None` if no index exists for the query's length or if no match is found.
    /// The returned `parent_idx` corresponds to the original input order.
    #[must_use]
    pub fn query(&self, seq: &[u8]) -> Option<MultiLenMatch> {
        let target_len = seq.len();

        // Binary search for the index with matching seq_len
        let len_group_idx = self
            .indices
            .binary_search_by_key(&target_len, |index| index.seq_len())
            .ok()?;

        self.indices[len_group_idx]
            .query(seq)
            .map(|inner| self.to_global_match(inner, len_group_idx, target_len))
    }

    /// Check if a sequence would be ambiguous (maps to multiple parents).
    ///
    /// Returns `false` if no index exists for the query's length.
    #[must_use]
    pub fn is_ambiguous(&self, seq: &[u8]) -> bool {
        let target_len = seq.len();

        let idx = match self
            .indices
            .binary_search_by_key(&target_len, |index| index.seq_len())
        {
            Ok(idx) => idx,
            Err(_) => return false,
        };

        self.indices[idx].is_ambiguous(seq)
    }

    /// Query at a specific position within a longer sequence.
    ///
    /// Checks all length-specific indices (in sorted order by length) and returns
    /// the first match found along with its length. The returned `parent_idx`
    /// corresponds to the original input order.
    ///
    /// # Example
    ///
    /// ```
    /// use seqhash::MultiLenSeqHash;
    ///
    /// let parents: Vec<&[u8]> = vec![b"ACGT", b"ACGTAC"];
    /// let index = MultiLenSeqHash::new(&parents).unwrap();
    ///
    /// let read = b"NNACGTACNN";
    /// let result = index.query_at(read, 2);
    /// // Returns first match (4 bp "ACGT" at position 2)
    /// assert!(result.is_some());
    /// assert_eq!(result.unwrap().seq_len(), 4);
    /// ```
    #[must_use]
    pub fn query_at(&self, seq: &[u8], pos: usize) -> Option<MultiLenMatch> {
        for (len_group_idx, index) in self.indices.iter().enumerate() {
            let seq_len = index.seq_len();
            let end = match pos.checked_add(seq_len) {
                Some(e) if e <= seq.len() => e,
                _ => continue,
            };

            if let Some(inner) = index.query(&seq[pos..end]) {
                return Some(self.to_global_match(inner, len_group_idx, seq_len));
            }
        }
        None
    }

    /// Query at a position with remapping window.
    ///
    /// For each offset (0, +1, -1, +2, -2, ...) up to `window`, checks all lengths
    /// and returns the first match found.
    #[must_use]
    pub fn query_at_with_remap(
        &self,
        seq: &[u8],
        pos: usize,
        window: usize,
    ) -> Option<MultiLenMatch> {
        self.query_at_with_remap_offset(seq, pos, window)
            .map(|(m, _)| m)
    }

    /// Query at a position with remapping, also returning the offset where match was found.
    #[must_use]
    pub fn query_at_with_remap_offset(
        &self,
        seq: &[u8],
        pos: usize,
        window: usize,
    ) -> Option<(MultiLenMatch, isize)> {
        // Try exact position first
        if let Some(m) = self.query_at(seq, pos) {
            return Some((m, 0));
        }

        // Alternate +1, -1, +2, -2, etc.
        for offset in 1..=window {
            if let Some(m) = self.query_at(seq, pos + offset) {
                return Some((m, offset as isize));
            }
            if offset <= pos {
                if let Some(m) = self.query_at(seq, pos - offset) {
                    return Some((m, -(offset as isize)));
                }
            }
        }

        None
    }

    /// Sliding window search from start of sequence.
    ///
    /// Scans through the sequence looking for the first match across all lengths.
    /// Returns the match and its position in the input sequence.
    #[must_use]
    pub fn query_sliding(&self, seq: &[u8]) -> Option<(MultiLenMatch, usize)> {
        self.query_sliding_iter(seq).next()
    }

    /// Sliding window search returning an iterator over all matches.
    ///
    /// Scans through the sequence and yields all matches found across all lengths.
    pub fn query_sliding_iter<'a>(
        &'a self,
        seq: &'a [u8],
    ) -> impl Iterator<Item = (MultiLenMatch, usize)> + 'a {
        let min_len = self.min_seq_len().unwrap_or(0);
        let num_positions = seq.len().saturating_sub(min_len.saturating_sub(1));

        (0..num_positions).filter_map(move |pos| self.query_at(seq, pos).map(|m| (m, pos)))
    }

    /// Get a parent sequence by global index.
    ///
    /// Returns `None` if the index is invalid.
    #[must_use]
    pub fn get_parent(&self, global_idx: usize) -> Option<&[u8]> {
        // Find which length group contains this global index
        for (len_group_idx, globals) in self.global_indices.iter().enumerate() {
            if let Some(local_idx) = globals.iter().position(|&g| g == global_idx) {
                return self.indices[len_group_idx].get_parent(local_idx);
            }
        }
        None
    }

    /// Returns the number of unique sequence lengths in the index.
    #[inline]
    #[must_use]
    pub fn num_lengths(&self) -> usize {
        self.indices.len()
    }

    /// Returns an iterator over all unique sequence lengths (sorted ascending).
    pub fn lengths(&self) -> impl Iterator<Item = usize> + '_ {
        self.indices.iter().map(|index| index.seq_len())
    }

    /// Returns the minimum sequence length, or `None` if empty.
    #[must_use]
    pub fn min_seq_len(&self) -> Option<usize> {
        self.indices.first().map(|index| index.seq_len())
    }

    /// Returns the maximum sequence length, or `None` if empty.
    #[must_use]
    pub fn max_seq_len(&self) -> Option<usize> {
        self.indices.last().map(|index| index.seq_len())
    }

    /// Returns the total number of parent sequences across all lengths.
    #[must_use]
    pub fn num_parents(&self) -> usize {
        self.indices.iter().map(|index| index.num_parents()).sum()
    }

    /// Returns the number of parent sequences for a specific length.
    ///
    /// Returns `None` if no index exists for the given length.
    #[must_use]
    pub fn num_parents_for_len(&self, seq_len: usize) -> Option<usize> {
        let idx = self
            .indices
            .binary_search_by_key(&seq_len, |index| index.seq_len())
            .ok()?;

        Some(self.indices[idx].num_parents())
    }

    /// Returns true if the index is empty (no sequences).
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }

    /// Check if a specific parent is within the specified hamming distance of the query sequence.
    ///
    /// The `global_idx` parameter is the global parent index from the original input.
    #[must_use]
    pub fn is_within_hdist(&self, query: &[u8], global_idx: usize, hdist: usize) -> bool {
        // Find which length group contains this global index
        for (len_group_idx, globals) in self.global_indices.iter().enumerate() {
            if let Some(local_idx) = globals.iter().position(|&g| g == global_idx) {
                return self.indices[len_group_idx].is_within_hdist(query, local_idx, hdist);
            }
        }
        false
    }

    /// Save the index to a file.
    ///
    /// The file will be saved in postcard format. The recommended extension is `.seqhash`.
    #[cfg(feature = "serde")]
    pub fn save<P: AsRef<std::path::Path>>(&self, path: P) -> std::io::Result<()> {
        crate::postcard_save(self, path.as_ref())
    }

    /// Load an index from a file.
    ///
    /// The file should be in postcard format, as created by [`MultiLenSeqHash::save`].
    #[cfg(feature = "serde")]
    pub fn load<P: AsRef<std::path::Path>>(path: P) -> std::io::Result<Self> {
        crate::postcard_load(path.as_ref())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // Construction tests
    // ========================================================================

    #[test]
    fn test_new_single_length() {
        let parents: Vec<&[u8]> = vec![b"ACGT", b"GGGG", b"TTTT"];
        let index = MultiLenSeqHash::new(&parents).unwrap();

        assert_eq!(index.num_lengths(), 1);
        assert_eq!(index.num_parents(), 3);
        assert_eq!(index.min_seq_len(), Some(4));
        assert_eq!(index.max_seq_len(), Some(4));
    }

    #[test]
    fn test_new_multiple_lengths() {
        let parents: Vec<&[u8]> = vec![
            b"ACGT",     // 4 bp, global 0
            b"GGGGCC",   // 6 bp, global 1
            b"TTTTAAAA", // 8 bp, global 2
            b"AAAA",     // 4 bp, global 3
        ];
        let index = MultiLenSeqHash::new(&parents).unwrap();

        assert_eq!(index.num_lengths(), 3);
        assert_eq!(index.num_parents(), 4);
        assert_eq!(index.min_seq_len(), Some(4));
        assert_eq!(index.max_seq_len(), Some(8));

        // Check per-length counts
        assert_eq!(index.num_parents_for_len(4), Some(2));
        assert_eq!(index.num_parents_for_len(6), Some(1));
        assert_eq!(index.num_parents_for_len(8), Some(1));
        assert_eq!(index.num_parents_for_len(5), None);
    }

    #[test]
    fn test_new_empty() {
        let parents: Vec<&[u8]> = vec![];
        let result = MultiLenSeqHash::new(&parents);
        assert!(matches!(result, Err(SeqHashError::EmptyParents)));
    }

    #[test]
    fn test_lengths_iterator() {
        let parents: Vec<&[u8]> = vec![
            b"ACGT",     // 4 bp
            b"GGGGCC",   // 6 bp
            b"TTTTAAAA", // 8 bp
        ];
        let index = MultiLenSeqHash::new(&parents).unwrap();

        let lengths: Vec<usize> = index.lengths().collect();
        assert_eq!(lengths, vec![4, 6, 8]);
    }

    #[test]
    fn test_builder_exact() {
        let parents: Vec<&[u8]> = vec![b"ACGT", b"GGGGCC"];
        let index = MultiLenSeqHashBuilder::default()
            .exact()
            .build(&parents)
            .unwrap();

        // Exact match should work
        assert!(index.query(b"ACGT").is_some());

        // Mismatch should not work with exact mode
        assert!(index.query(b"NCGT").is_none());
    }

    #[test]
    fn test_builder_exclude_n() {
        let parents: Vec<&[u8]> = vec![b"ACGN"];
        let result = MultiLenSeqHashBuilder::default()
            .exclude_n()
            .build(&parents);

        assert!(matches!(result, Err(SeqHashError::InvalidBase { .. })));
    }

    // ========================================================================
    // Global index tests - CRITICAL
    // ========================================================================

    #[test]
    fn test_global_index_preservation() {
        // This is the critical test - indices must match original input order
        let parents: Vec<&[u8]> = vec![
            b"ACGT",   // global 0, 4 bp
            b"GGGGCC", // global 1, 6 bp
            b"TTTT",   // global 2, 4 bp
            b"AAAA",   // global 3, 4 bp
            b"CCCCCC", // global 4, 6 bp
        ];
        let index = MultiLenSeqHash::new(&parents).unwrap();

        // Query each parent and verify global index
        let result = index.query(b"ACGT").unwrap();
        assert_eq!(result.parent_idx(), 0);

        let result = index.query(b"GGGGCC").unwrap();
        assert_eq!(result.parent_idx(), 1);

        let result = index.query(b"TTTT").unwrap();
        assert_eq!(result.parent_idx(), 2);

        let result = index.query(b"AAAA").unwrap();
        assert_eq!(result.parent_idx(), 3);

        let result = index.query(b"CCCCCC").unwrap();
        assert_eq!(result.parent_idx(), 4);
    }

    #[test]
    fn test_global_index_with_mismatch() {
        let parents: Vec<&[u8]> = vec![
            b"ACGT",   // global 0, 4 bp
            b"GGGGCC", // global 1, 6 bp
            b"TTTT",   // global 2, 4 bp
        ];
        let index = MultiLenSeqHash::new(&parents).unwrap();

        // Query with one mismatch - should still return correct global index
        let result = index.query(b"NCGT").unwrap(); // mismatch from ACGT
        assert_eq!(result.parent_idx(), 0);
        assert!(!result.is_exact());

        let result = index.query(b"TTTA").unwrap(); // mismatch from TTTT
        assert_eq!(result.parent_idx(), 2);
        assert!(!result.is_exact());
    }

    #[test]
    fn test_global_index_query_at() {
        let parents: Vec<&[u8]> = vec![
            b"ACGT",   // global 0, 4 bp
            b"GGGGCC", // global 1, 6 bp
            b"TTTT",   // global 2, 4 bp
        ];
        let index = MultiLenSeqHash::new(&parents).unwrap();

        // query_at should return correct global index
        let read = b"NNTTTTNN";
        let result = index.query_at(read, 2).unwrap();
        assert_eq!(result.parent_idx(), 2); // TTTT is global index 2
        assert_eq!(result.seq_len(), 4);
    }

    #[test]
    fn test_global_index_query_sliding() {
        let parents: Vec<&[u8]> = vec![
            b"AAAA",   // global 0, 4 bp
            b"GGGGCC", // global 1, 6 bp
            b"TTTT",   // global 2, 4 bp
        ];
        let index = MultiLenSeqHash::new(&parents).unwrap();

        // Verify direct queries first
        assert_eq!(index.query(b"AAAA").unwrap().parent_idx(), 0);
        assert_eq!(index.query(b"GGGGCC").unwrap().parent_idx(), 1);
        assert_eq!(index.query(b"TTTT").unwrap().parent_idx(), 2);

        // Test query_at directly at position 2
        let read = b"NNTTTTNN";
        let result = index.query_at(read, 2).unwrap();
        assert_eq!(
            result.parent_idx(),
            2,
            "query_at(2) should return TTTT (global 2)"
        );

        // Test sliding - first match is at position 1 (NTTT matches TTTT with 1 mismatch)
        let (result, pos) = index.query_sliding(read).unwrap();
        assert_eq!(result.parent_idx(), 2); // All matches are TTTT (global index 2)
        assert_eq!(pos, 1); // First match at position 1 (1-mismatch)
        assert!(!result.is_exact()); // It's a mismatch match
    }

    #[test]
    fn test_get_parent_by_global_index() {
        let parents: Vec<&[u8]> = vec![
            b"ACGT",   // global 0
            b"GGGGCC", // global 1
            b"TTTT",   // global 2
            b"AAAA",   // global 3
        ];
        let index = MultiLenSeqHash::new(&parents).unwrap();

        assert_eq!(index.get_parent(0), Some(b"ACGT".as_slice()));
        assert_eq!(index.get_parent(1), Some(b"GGGGCC".as_slice()));
        assert_eq!(index.get_parent(2), Some(b"TTTT".as_slice()));
        assert_eq!(index.get_parent(3), Some(b"AAAA".as_slice()));
        assert_eq!(index.get_parent(4), None);
    }

    #[test]
    fn test_is_within_hdist_global_index() {
        let parents: Vec<&[u8]> = vec![
            b"ACGT",   // global 0
            b"GGGGCC", // global 1
            b"TTTT",   // global 2
        ];
        let index = MultiLenSeqHash::new(&parents).unwrap();

        // Check using global indices
        assert!(index.is_within_hdist(b"ACGT", 0, 0)); // exact match
        assert!(index.is_within_hdist(b"NCGT", 0, 1)); // 1 mismatch
        assert!(!index.is_within_hdist(b"NNGT", 0, 1)); // 2 mismatches

        assert!(index.is_within_hdist(b"TTTT", 2, 0)); // global index 2
        assert!(index.is_within_hdist(b"TTTA", 2, 1));
    }

    // ========================================================================
    // Query tests
    // ========================================================================

    #[test]
    fn test_query_exact_match() {
        let parents: Vec<&[u8]> = vec![
            b"ACGT",     // 4 bp
            b"GGGGCC",   // 6 bp
            b"TTTTAAAA", // 8 bp
        ];
        let index = MultiLenSeqHash::new(&parents).unwrap();

        // Query 4 bp
        let result = index.query(b"ACGT");
        assert!(result.is_some());
        let m = result.unwrap();
        assert_eq!(m.seq_len(), 4);
        assert!(m.is_exact());

        // Query 6 bp
        let result = index.query(b"GGGGCC");
        assert!(result.is_some());
        let m = result.unwrap();
        assert_eq!(m.seq_len(), 6);

        // Query 8 bp
        let result = index.query(b"TTTTAAAA");
        assert!(result.is_some());
        let m = result.unwrap();
        assert_eq!(m.seq_len(), 8);
    }

    #[test]
    fn test_query_mismatch() {
        let parents: Vec<&[u8]> = vec![b"ACGT", b"GGGGCC"];
        let index = MultiLenSeqHash::new(&parents).unwrap();

        // One mismatch
        let result = index.query(b"NCGT");
        assert!(result.is_some());
        let m = result.unwrap();
        assert_eq!(m.seq_len(), 4);
        assert!(!m.is_exact());
        assert_eq!(m.mismatch_pos(), Some(0));
    }

    #[test]
    fn test_query_no_match_wrong_length() {
        let parents: Vec<&[u8]> = vec![b"ACGT", b"GGGGCC"];
        let index = MultiLenSeqHash::new(&parents).unwrap();

        // Query with length that doesn't exist
        assert!(index.query(b"ACGTA").is_none()); // 5 bp doesn't exist
    }

    #[test]
    fn test_query_no_match_not_found() {
        let parents: Vec<&[u8]> = vec![b"ACGT"];
        let index = MultiLenSeqHash::new(&parents).unwrap();

        // Two mismatches - should not match
        assert!(index.query(b"NNGT").is_none());
    }

    #[test]
    fn test_is_ambiguous() {
        // Two parents one mismatch apart
        let parents: Vec<&[u8]> = vec![b"ACGT", b"TCGT"];
        let index = MultiLenSeqHash::new(&parents).unwrap();

        // CCGT is one mismatch from both
        assert!(index.is_ambiguous(b"CCGT"));

        // Wrong length returns false
        assert!(!index.is_ambiguous(b"CCGTAA"));
    }

    // ========================================================================
    // Positional query tests
    // ========================================================================

    #[test]
    fn test_query_at_single_length() {
        let parents: Vec<&[u8]> = vec![b"ACGT"];
        let index = MultiLenSeqHash::new(&parents).unwrap();

        let read = b"NNACGTNN";
        let result = index.query_at(read, 2);
        assert!(result.is_some());
        let m = result.unwrap();
        assert_eq!(m.seq_len(), 4);
        assert!(m.is_exact());
    }

    #[test]
    fn test_query_at_multiple_lengths() {
        let parents: Vec<&[u8]> = vec![
            b"ACGT",   // 4 bp
            b"ACGTAC", // 6 bp
        ];
        let index = MultiLenSeqHash::new(&parents).unwrap();

        // Both could match at position 2, but shorter (4 bp) is checked first
        let read = b"NNACGTACNN";
        let result = index.query_at(read, 2);
        assert!(result.is_some());
        let m = result.unwrap();
        assert_eq!(m.seq_len(), 4); // First match wins (sorted by length, smallest first)
    }

    #[test]
    fn test_query_at_only_longer_matches() {
        let parents: Vec<&[u8]> = vec![
            b"TTTT",   // 4 bp - won't match at position 2
            b"ACGTAC", // 6 bp - will match
        ];
        let index = MultiLenSeqHash::new(&parents).unwrap();

        let read = b"NNACGTACNN";
        let result = index.query_at(read, 2);
        assert!(result.is_some());
        let m = result.unwrap();
        assert_eq!(m.seq_len(), 6);
    }

    #[test]
    fn test_query_at_out_of_bounds() {
        let parents: Vec<&[u8]> = vec![b"ACGT"];
        let index = MultiLenSeqHash::new(&parents).unwrap();

        let read = b"NNNN";
        assert!(index.query_at(read, 2).is_none());
    }

    #[test]
    fn test_query_at_with_remap() {
        let parents: Vec<&[u8]> = vec![b"ACGT"];
        let index = MultiLenSeqHash::new(&parents).unwrap();

        // Target at position 3, looking from position 2
        let read = b"NNNACGTNN";
        let result = index.query_at_with_remap(read, 2, 3);
        assert!(result.is_some());
        assert_eq!(result.unwrap().seq_len(), 4);
    }

    #[test]
    fn test_query_at_with_remap_offset() {
        let parents: Vec<&[u8]> = vec![b"ACGT"];
        let index = MultiLenSeqHash::new(&parents).unwrap();

        // Target at position 4, looking from position 2
        let read = b"NNNNACGTNN";
        let result = index.query_at_with_remap_offset(read, 2, 3);
        assert!(result.is_some());
        let (m, offset) = result.unwrap();
        assert_eq!(m.seq_len(), 4);
        assert_eq!(offset, 2);
    }

    #[test]
    fn test_query_sliding() {
        let parents: Vec<&[u8]> = vec![b"ACGT", b"GGGGCC"];
        let index = MultiLenSeqHash::new(&parents).unwrap();

        let read = b"NNNACGTNNN";
        let result = index.query_sliding(read);
        assert!(result.is_some());
        let (m, pos) = result.unwrap();
        assert_eq!(m.seq_len(), 4);
        assert_eq!(pos, 3);
    }

    #[test]
    fn test_query_sliding_iter() {
        let parents: Vec<&[u8]> = vec![b"ACGT"];
        let index = MultiLenSeqHash::new(&parents).unwrap();

        // Two occurrences
        let read = b"ACGTNNACGT";
        let matches: Vec<_> = index.query_sliding_iter(read).collect();

        assert_eq!(matches.len(), 2);
        assert_eq!(matches[0].1, 0);
        assert_eq!(matches[1].1, 6);
    }

    // ========================================================================
    // Accessor tests
    // ========================================================================

    #[test]
    fn test_is_empty() {
        let parents: Vec<&[u8]> = vec![b"ACGT"];
        let index = MultiLenSeqHash::new(&parents).unwrap();
        assert!(!index.is_empty());

        let empty = MultiLenSeqHash::default();
        assert!(empty.is_empty());
    }

    // ========================================================================
    // MultiLenMatch tests
    // ========================================================================

    #[test]
    fn test_multilen_match_methods() {
        let m = MultiLenMatch {
            parent_idx: 5,
            mismatch_pos: None,
            seq_len: 10,
        };

        assert_eq!(m.parent_idx(), 5);
        assert!(m.is_exact());
        assert_eq!(m.mismatch_pos(), None);
        assert_eq!(m.hdist(), 0);
        assert_eq!(m.seq_len(), 10);

        let m = MultiLenMatch {
            parent_idx: 3,
            mismatch_pos: Some(7),
            seq_len: 12,
        };

        assert_eq!(m.parent_idx(), 3);
        assert!(!m.is_exact());
        assert_eq!(m.mismatch_pos(), Some(7));
        assert_eq!(m.hdist(), 1);
        assert_eq!(m.seq_len(), 12);
    }
}

#[cfg(all(test, feature = "serde"))]
mod serde_tests {
    use super::*;

    #[test]
    fn test_multilen_seqhash_serde_json() {
        let parents: Vec<&[u8]> = vec![b"ACGT", b"GGGGCC", b"TTTTAAAA"];

        let index = MultiLenSeqHash::new(&parents).unwrap();

        // Serialize to JSON
        let json = serde_json::to_string(&index).unwrap();

        // Deserialize from JSON
        let restored: MultiLenSeqHash = serde_json::from_str(&json).unwrap();

        // Verify structure
        assert_eq!(restored.num_lengths(), index.num_lengths());
        assert_eq!(restored.num_parents(), index.num_parents());

        // Verify queries return correct global indices
        let result = restored.query(b"ACGT").unwrap();
        assert_eq!(result.parent_idx(), 0);

        let result = restored.query(b"GGGGCC").unwrap();
        assert_eq!(result.parent_idx(), 1);

        let result = restored.query(b"TTTTAAAA").unwrap();
        assert_eq!(result.parent_idx(), 2);
    }
}

#[cfg(all(test, feature = "serde"))]
mod persistence_tests {
    use super::*;

    #[test]
    fn test_multilen_seqhash_roundtrip_postcard() {
        let parents: Vec<&[u8]> = vec![b"ACGT", b"GGGGCC", b"TTTTAAAA"];

        let index = MultiLenSeqHash::new(&parents).unwrap();

        let bytes = postcard::to_stdvec(&index).unwrap();
        let restored: MultiLenSeqHash = postcard::from_bytes(&bytes).unwrap();

        // Verify structure
        assert_eq!(restored.num_lengths(), index.num_lengths());
        assert_eq!(restored.num_parents(), index.num_parents());

        // Verify queries return correct global indices
        for (global_idx, parent) in parents.iter().enumerate() {
            let result = restored.query(parent).unwrap();
            assert_eq!(result.parent_idx(), global_idx);
        }
    }

    #[test]
    fn test_multilen_save_and_load() {
        let parents: Vec<&[u8]> = vec![b"ACGT", b"GGGGCC", b"TTTTAAAA"];
        let index = MultiLenSeqHash::new(&parents).unwrap();

        let temp_dir = std::env::temp_dir();
        let file_path = temp_dir.join("test_multilen_index.seqhash");

        index.save(&file_path).unwrap();
        let loaded = MultiLenSeqHash::load(&file_path).unwrap();

        assert_eq!(loaded.num_lengths(), index.num_lengths());
        assert_eq!(loaded.num_parents(), index.num_parents());
        for (global_idx, parent) in parents.iter().enumerate() {
            let result = loaded.query(parent).unwrap();
            assert_eq!(result.parent_idx(), global_idx);
        }

        std::fs::remove_file(&file_path).ok();
    }
}
