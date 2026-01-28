//! Split-map sequence matching with higher mismatch tolerance.
//!
//! This module provides [`SplitSeqHash`], which divides sequences in half and
//! maintains separate [`crate::SeqHash`] indices for each half. This enables
//! matching strategies that tolerate more total mismatches while keeping
//! lookups fast.

use crate::{Match, SeqHash, SeqHashError};

/// Which half of a split sequence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Half {
    /// Left half of the sequence.
    Left,
    /// Right half of the sequence.
    Right,
}

impl Half {
    /// Returns the other half.
    #[inline]
    #[must_use]
    pub fn other(&self) -> Half {
        match self {
            Half::Left => Half::Right,
            Half::Right => Half::Left,
        }
    }
}

/// Result of querying both halves of a sequence.
#[derive(Debug, Clone)]
pub struct SplitMatch {
    /// Match result for the left half.
    pub left: Option<Match>,
    /// Match result for the right half.
    pub right: Option<Match>,
}

impl SplitMatch {
    /// Returns true if at least one half matched.
    ///
    /// This is useful for determining if a query found anything at all,
    /// even if the match is partial or conflicted.
    #[inline]
    #[must_use]
    pub fn has_match(&self) -> bool {
        self.left.is_some() || self.right.is_some()
    }

    /// Returns the parent index if both halves matched and agree on the same parent.
    ///
    /// This is the "happy path" - both halves found a match (exact or 1-mismatch)
    /// and they point to the same parent sequence.
    #[must_use]
    pub fn agreed_idx(&self) -> Option<usize> {
        match (&self.left, &self.right) {
            (Some(left), Some(right)) => {
                let left_idx = left.parent_idx();
                let right_idx = right.parent_idx();
                if left_idx == right_idx {
                    Some(left_idx)
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// Returns (parent_idx, which_half) if exactly one half matched.
    ///
    /// Useful for fallback logic where you want to validate the non-matching
    /// half using hamming distance.
    #[must_use]
    pub fn single_match(&self) -> Option<(usize, Half)> {
        match (&self.left, &self.right) {
            (Some(left), None) => Some((left.parent_idx(), Half::Left)),
            (None, Some(right)) => Some((right.parent_idx(), Half::Right)),
            _ => None,
        }
    }

    /// Returns true if both halves matched but to different parents.
    ///
    /// This indicates an ambiguous/conflicting result that should be rejected.
    #[must_use]
    pub fn is_conflicted(&self) -> bool {
        match (&self.left, &self.right) {
            (Some(left), Some(right)) => left.parent_idx() != right.parent_idx(),
            _ => false,
        }
    }

    /// Returns the total hamming distance from the matched portions.
    ///
    /// - Exact match contributes 0
    /// - Mismatch match contributes 1
    /// - Returns None if neither half matched
    ///
    /// This represents the "budget spent" by the SeqHash matching.
    #[must_use]
    pub fn matched_hdist(&self) -> Option<usize> {
        match (&self.left, &self.right) {
            (None, None) => None,
            (Some(m), None) | (None, Some(m)) => Some(m.hdist()),
            (Some(left), Some(right)) => Some(left.hdist() + right.hdist()),
        }
    }

    /// Returns the remaining hdist budget after accounting for matched portions.
    ///
    /// Given a maximum allowed hamming distance, returns how much tolerance
    /// remains for validating unmatched portions.
    ///
    /// # Example
    ///
    /// If `max_hdist = 3` and the left half matched with a mismatch (hdist=1),
    /// then `remaining_hdist(3)` returns `Some(2)`.
    #[must_use]
    pub fn remaining_hdist(&self, max_hdist: usize) -> Option<usize> {
        self.matched_hdist()
            .and_then(|used| max_hdist.checked_sub(used))
    }
}

/// A split-map sequence index for higher mismatch tolerance.
///
/// Divides sequences in half and maintains separate [`SeqHash`] indices for each half.
/// This enables matching strategies that tolerate more total mismatches by:
/// 1. Using fast SeqHash lookups (≤1 mismatch) on each half
/// 2. Requiring both halves to agree on the parent
/// 3. Falling back to hamming distance validation when one half fails
///
/// # Example
///
/// ```
/// use seqhash::SplitSeqHash;
///
/// let parents: Vec<&[u8]> = vec![
///     b"ACGTACGTACGTACGT",
///     b"GGGGCCCCGGGGCCCC",
/// ];
///
/// let index = SplitSeqHash::new(&parents).unwrap();
///
/// // Query a sequence
/// let result = index.query(b"ACGTACGTACGTACGT");
///
/// // Both halves matched the same parent
/// assert_eq!(result.agreed_idx(), Some(0));
/// ```
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SplitSeqHash {
    left: SeqHash,
    right: SeqHash,
    split_pos: usize,
    seq_len: usize,
    num_parents: usize,
}

impl SplitSeqHash {
    /// Create a new split index, dividing sequences at the midpoint.
    ///
    /// All parent sequences must have the same length. The split position
    /// is `seq_len / 2`.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Parents have inconsistent lengths
    /// - Parents slice is empty
    /// - SeqHash construction fails for either half
    pub fn new<S: AsRef<[u8]>>(parents: &[S]) -> Result<Self, SeqHashError> {
        if parents.is_empty() {
            return Err(SeqHashError::EmptyParents);
        }

        let seq_len = parents[0].as_ref().len();
        let split_pos = seq_len / 2;

        Self::with_split_pos(parents, split_pos)
    }

    /// Create a new split index with an explicit split position.
    ///
    /// # Arguments
    ///
    /// * `parents` - Slice of parent sequences (all must have same length)
    /// * `split_pos` - Position to split at (left half is `[0..split_pos]`)
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - `split_pos` is 0 or >= seq_len
    /// - Parents have inconsistent lengths
    /// - Parents slice is empty
    /// - SeqHash construction fails for either half
    pub fn with_split_pos<S: AsRef<[u8]>>(
        parents: &[S],
        split_pos: usize,
    ) -> Result<Self, SeqHashError> {
        if parents.is_empty() {
            return Err(SeqHashError::EmptyParents);
        }

        let seq_len = parents[0].as_ref().len();

        // Validate split position
        if split_pos == 0 || split_pos >= seq_len {
            return Err(SeqHashError::InconsistentLength {
                expected: seq_len,
                found: split_pos,
                index: 0,
            });
        }

        // Check all parents have the same length
        for (i, parent) in parents.iter().enumerate() {
            let parent_len = parent.as_ref().len();
            if parent_len != seq_len {
                return Err(SeqHashError::InconsistentLength {
                    expected: seq_len,
                    found: parent_len,
                    index: i,
                });
            }
        }

        // Extract left and right halves
        let left_seqs: Vec<Vec<u8>> = parents
            .iter()
            .map(|p| p.as_ref()[..split_pos].to_vec())
            .collect();

        let right_seqs: Vec<Vec<u8>> = parents
            .iter()
            .map(|p| p.as_ref()[split_pos..].to_vec())
            .collect();

        // Build SeqHash indices for each half
        let left = SeqHash::new(&left_seqs)?;
        let right = SeqHash::new(&right_seqs)?;

        Ok(Self {
            left,
            right,
            split_pos,
            seq_len,
            num_parents: parents.len(),
        })
    }

    /// Query both halves of a sequence.
    ///
    /// Splits the query at `split_pos` and queries each half against its
    /// respective [`SeqHash`] index. Returns a [`SplitMatch`] containing both
    /// results.
    ///
    /// # Panics
    ///
    /// Panics if `seq.len() != self.seq_len()`.
    #[must_use]
    pub fn query(&self, seq: &[u8]) -> SplitMatch {
        assert_eq!(
            seq.len(),
            self.seq_len,
            "Query sequence length {} does not match expected length {}",
            seq.len(),
            self.seq_len
        );

        let left_query = &seq[..self.split_pos];
        let right_query = &seq[self.split_pos..];

        let left = self.left.query(left_query);
        let right = self.right.query(right_query);

        SplitMatch { left, right }
    }

    /// Check if a half of the sequence is within hamming distance of a parent's corresponding half.
    ///
    /// This is used for fallback validation when only one half matched via [`query()`](Self::query).
    /// The sequence is split internally; only the specified half is checked.
    ///
    /// # Arguments
    ///
    /// * `seq` - Full query sequence (will be split internally)
    /// * `parent_idx` - Index of the parent to compare against
    /// * `half` - Which half to check
    /// * `max_hdist` - Maximum allowed hamming distance
    ///
    /// # Returns
    ///
    /// Returns `false` if:
    /// - `seq.len() != self.seq_len()`
    /// - `parent_idx` is out of bounds
    /// - The hamming distance exceeds `max_hdist`
    #[must_use]
    pub fn is_within_hdist(
        &self,
        seq: &[u8],
        parent_idx: usize,
        half: Half,
        max_hdist: usize,
    ) -> bool {
        if seq.len() != self.seq_len {
            return false;
        }

        if parent_idx >= self.num_parents {
            return false;
        }

        match half {
            Half::Left => {
                let query_half = &seq[..self.split_pos];
                self.left.is_within_hdist(query_half, parent_idx, max_hdist)
            }
            Half::Right => {
                let query_half = &seq[self.split_pos..];
                self.right
                    .is_within_hdist(query_half, parent_idx, max_hdist)
            }
        }
    }

    /// Query at a specific position within a longer sequence.
    ///
    /// Extracts a subsequence of length `seq_len` starting at `pos` and queries it.
    /// Returns a `SplitMatch` with `None` for both halves if the position is out of bounds.
    ///
    /// # Example
    ///
    /// ```
    /// use seqhash::SplitSeqHash;
    ///
    /// let parents: Vec<&[u8]> = vec![b"ACGTACGT"];
    /// let index = SplitSeqHash::new(&parents).unwrap();
    ///
    /// // Target sequence is embedded at position 2
    /// let read = b"NNACGTACGTNN";
    /// let result = index.query_at(read, 2);
    /// assert_eq!(result.agreed_idx(), Some(0));
    /// ```
    #[inline]
    #[must_use]
    pub fn query_at(&self, seq: &[u8], pos: usize) -> SplitMatch {
        let end = match pos.checked_add(self.seq_len) {
            Some(e) if e <= seq.len() => e,
            _ => {
                return SplitMatch {
                    left: None,
                    right: None,
                }
            }
        };
        self.query(&seq[pos..end])
    }

    /// Query at a position with remapping window.
    ///
    /// Tries `pos` first, then alternates `+1, -1, +2, -2, ...` up to `window`.
    /// Returns the first `SplitMatch` where at least one half matched,
    /// or a no-match `SplitMatch` if nothing found within the window.
    ///
    /// This is useful when the target position may drift slightly due to small indels.
    ///
    /// # Example
    ///
    /// ```
    /// use seqhash::SplitSeqHash;
    ///
    /// let parents: Vec<&[u8]> = vec![b"ACGTACGT"];
    /// let index = SplitSeqHash::new(&parents).unwrap();
    ///
    /// // Target is at position 3, but we think it's at position 2
    /// let read = b"NNNACGTACGTNN";
    /// let result = index.query_at_with_remap(read, 2, 2);
    /// assert_eq!(result.agreed_idx(), Some(0));
    /// ```
    #[inline]
    #[must_use]
    pub fn query_at_with_remap(&self, seq: &[u8], pos: usize, window: usize) -> SplitMatch {
        self.query_at_with_remap_offset(seq, pos, window).0
    }

    /// Query at a position with remapping, also returning the offset where match was found.
    ///
    /// Tries `pos` first, then alternates `+1, -1, +2, -2, ...` up to `window`.
    /// Returns the `SplitMatch` and offset (0 for direct hit, positive for downstream,
    /// negative for upstream).
    ///
    /// A match is considered "found" when at least one half matches. This allows
    /// for fallback validation of the non-matching half using [`is_within_hdist`](Self::is_within_hdist).
    ///
    /// # Example
    ///
    /// ```
    /// use seqhash::SplitSeqHash;
    ///
    /// let parents: Vec<&[u8]> = vec![b"ACGTACGT"];
    /// let index = SplitSeqHash::new(&parents).unwrap();
    ///
    /// // Target is at position 3, but we think it's at position 2
    /// let read = b"NNNACGTACGTNN";
    /// let (result, offset) = index.query_at_with_remap_offset(read, 2, 2);
    /// assert_eq!(result.agreed_idx(), Some(0));
    /// assert_eq!(offset, 1); // Found at offset +1
    /// ```
    #[must_use]
    pub fn query_at_with_remap_offset(
        &self,
        seq: &[u8],
        pos: usize,
        window: usize,
    ) -> (SplitMatch, isize) {
        // Try exact position first
        let result = self.query_at(seq, pos);
        if result.has_match() {
            return (result, 0);
        }

        // Alternate +1, -1, +2, -2, etc.
        for offset in 1..=window {
            // Try positive offset
            if let Some(try_pos) = pos.checked_add(offset) {
                let result = self.query_at(seq, try_pos);
                if result.has_match() {
                    return (result, offset as isize);
                }
            }

            // Try negative offset
            if let Some(try_pos) = pos.checked_sub(offset) {
                let result = self.query_at(seq, try_pos);
                if result.has_match() {
                    return (result, -(offset as isize));
                }
            }
        }

        (
            SplitMatch {
                left: None,
                right: None,
            },
            0,
        )
    }

    /// Sliding window search from start of sequence.
    ///
    /// Scans through the sequence looking for the first position where at least one half matches.
    /// Returns the `SplitMatch` and its position in the input sequence.
    #[must_use]
    pub fn query_sliding(&self, seq: &[u8]) -> Option<(SplitMatch, usize)> {
        self.query_sliding_iter(seq).next()
    }

    /// Sliding window search returning an iterator over all matches.
    ///
    /// Scans through the sequence and yields all positions where at least one half matches.
    pub fn query_sliding_iter<'a>(
        &'a self,
        seq: &'a [u8],
    ) -> impl Iterator<Item = (SplitMatch, usize)> + 'a {
        let num_positions = seq.len().saturating_sub(self.seq_len - 1);
        (0..num_positions).filter_map(move |pos| {
            let result = self.query_at(seq, pos);
            if result.has_match() {
                Some((result, pos))
            } else {
                None
            }
        })
    }

    /// Returns the full sequence length.
    #[inline]
    #[must_use]
    pub fn seq_len(&self) -> usize {
        self.seq_len
    }

    /// Returns the split position (length of left half).
    #[inline]
    #[must_use]
    pub fn split_pos(&self) -> usize {
        self.split_pos
    }

    /// Returns the number of parent sequences.
    #[inline]
    #[must_use]
    pub fn num_parents(&self) -> usize {
        self.num_parents
    }

    /// Returns the total number of entries across both halves.
    #[inline]
    #[must_use]
    pub fn num_entries(&self) -> usize {
        self.left.num_entries() + self.right.num_entries()
    }

    /// Returns the length of the left half.
    #[inline]
    #[must_use]
    pub fn left_len(&self) -> usize {
        self.split_pos
    }

    /// Returns the length of the right half.
    #[inline]
    #[must_use]
    pub fn right_len(&self) -> usize {
        self.seq_len - self.split_pos
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Half;

    // ========================================================================
    // Half tests
    // ========================================================================

    #[test]
    fn test_half_other() {
        assert_eq!(Half::Left.other(), Half::Right);
        assert_eq!(Half::Right.other(), Half::Left);
    }

    // ========================================================================
    // SplitMatch tests
    // ========================================================================

    #[test]
    fn test_split_match_agreed_idx_both_exact() {
        let result = SplitMatch {
            left: Some(Match::Exact { parent_idx: 0 }),
            right: Some(Match::Exact { parent_idx: 0 }),
        };
        assert_eq!(result.agreed_idx(), Some(0));
    }

    #[test]
    fn test_split_match_agreed_idx_both_mismatch() {
        let result = SplitMatch {
            left: Some(Match::Mismatch {
                parent_idx: 1,
                pos: 2,
            }),
            right: Some(Match::Mismatch {
                parent_idx: 1,
                pos: 5,
            }),
        };
        assert_eq!(result.agreed_idx(), Some(1));
    }

    #[test]
    fn test_split_match_agreed_idx_mixed() {
        let result = SplitMatch {
            left: Some(Match::Exact { parent_idx: 2 }),
            right: Some(Match::Mismatch {
                parent_idx: 2,
                pos: 3,
            }),
        };
        assert_eq!(result.agreed_idx(), Some(2));
    }

    #[test]
    fn test_split_match_agreed_idx_disagreement() {
        let result = SplitMatch {
            left: Some(Match::Exact { parent_idx: 0 }),
            right: Some(Match::Exact { parent_idx: 1 }),
        };
        assert_eq!(result.agreed_idx(), None);
    }

    #[test]
    fn test_split_match_agreed_idx_only_left() {
        let result = SplitMatch {
            left: Some(Match::Exact { parent_idx: 0 }),
            right: None,
        };
        assert_eq!(result.agreed_idx(), None);
    }

    #[test]
    fn test_split_match_agreed_idx_only_right() {
        let result = SplitMatch {
            left: None,
            right: Some(Match::Exact { parent_idx: 0 }),
        };
        assert_eq!(result.agreed_idx(), None);
    }

    #[test]
    fn test_split_match_agreed_idx_neither() {
        let result = SplitMatch {
            left: None,
            right: None,
        };
        assert_eq!(result.agreed_idx(), None);
    }

    #[test]
    fn test_split_match_single_match_left() {
        let result = SplitMatch {
            left: Some(Match::Exact { parent_idx: 0 }),
            right: None,
        };
        assert_eq!(result.single_match(), Some((0, Half::Left)));
    }

    #[test]
    fn test_split_match_single_match_right() {
        let result = SplitMatch {
            left: None,
            right: Some(Match::Mismatch {
                parent_idx: 1,
                pos: 3,
            }),
        };
        assert_eq!(result.single_match(), Some((1, Half::Right)));
    }

    #[test]
    fn test_split_match_single_match_both() {
        let result = SplitMatch {
            left: Some(Match::Exact { parent_idx: 0 }),
            right: Some(Match::Exact { parent_idx: 0 }),
        };
        assert_eq!(result.single_match(), None);
    }

    #[test]
    fn test_split_match_single_match_neither() {
        let result = SplitMatch {
            left: None,
            right: None,
        };
        assert_eq!(result.single_match(), None);
    }

    #[test]
    fn test_split_match_is_conflicted_true() {
        let result = SplitMatch {
            left: Some(Match::Exact { parent_idx: 0 }),
            right: Some(Match::Exact { parent_idx: 1 }),
        };
        assert!(result.is_conflicted());
    }

    #[test]
    fn test_split_match_is_conflicted_false_same() {
        let result = SplitMatch {
            left: Some(Match::Exact { parent_idx: 0 }),
            right: Some(Match::Exact { parent_idx: 0 }),
        };
        assert!(!result.is_conflicted());
    }

    #[test]
    fn test_split_match_is_conflicted_false_partial() {
        let result = SplitMatch {
            left: Some(Match::Exact { parent_idx: 0 }),
            right: None,
        };
        assert!(!result.is_conflicted());
    }

    #[test]
    fn test_split_match_is_conflicted_false_neither() {
        let result = SplitMatch {
            left: None,
            right: None,
        };
        assert!(!result.is_conflicted());
    }

    #[test]
    fn test_split_match_matched_hdist_both_exact() {
        let result = SplitMatch {
            left: Some(Match::Exact { parent_idx: 0 }),
            right: Some(Match::Exact { parent_idx: 0 }),
        };
        assert_eq!(result.matched_hdist(), Some(0));
    }

    #[test]
    fn test_split_match_matched_hdist_one_mismatch() {
        let result = SplitMatch {
            left: Some(Match::Mismatch {
                parent_idx: 0,
                pos: 2,
            }),
            right: Some(Match::Exact { parent_idx: 0 }),
        };
        assert_eq!(result.matched_hdist(), Some(1));
    }

    #[test]
    fn test_split_match_matched_hdist_both_mismatch() {
        let result = SplitMatch {
            left: Some(Match::Mismatch {
                parent_idx: 0,
                pos: 2,
            }),
            right: Some(Match::Mismatch {
                parent_idx: 0,
                pos: 5,
            }),
        };
        assert_eq!(result.matched_hdist(), Some(2));
    }

    #[test]
    fn test_split_match_matched_hdist_only_left() {
        let result = SplitMatch {
            left: Some(Match::Mismatch {
                parent_idx: 0,
                pos: 2,
            }),
            right: None,
        };
        assert_eq!(result.matched_hdist(), Some(1));
    }

    #[test]
    fn test_split_match_matched_hdist_only_right() {
        let result = SplitMatch {
            left: None,
            right: Some(Match::Exact { parent_idx: 0 }),
        };
        assert_eq!(result.matched_hdist(), Some(0));
    }

    #[test]
    fn test_split_match_matched_hdist_neither() {
        let result = SplitMatch {
            left: None,
            right: None,
        };
        assert_eq!(result.matched_hdist(), None);
    }

    #[test]
    fn test_split_match_remaining_hdist() {
        let result = SplitMatch {
            left: Some(Match::Mismatch {
                parent_idx: 0,
                pos: 2,
            }),
            right: Some(Match::Exact { parent_idx: 0 }),
        };
        assert_eq!(result.remaining_hdist(3), Some(2));
        assert_eq!(result.remaining_hdist(1), Some(0));
        assert_eq!(result.remaining_hdist(0), None);
    }

    #[test]
    fn test_split_match_remaining_hdist_no_match() {
        let result = SplitMatch {
            left: None,
            right: None,
        };
        assert_eq!(result.remaining_hdist(3), None);
    }

    // ========================================================================
    // SplitSeqHash tests
    // ========================================================================

    #[test]
    fn test_split_seqhash_new() {
        let parents: Vec<&[u8]> = vec![b"ACGTACGTACGTACGT", b"GGGGCCCCGGGGCCCC"];

        let index = SplitSeqHash::new(&parents).unwrap();

        assert_eq!(index.seq_len(), 16);
        assert_eq!(index.split_pos(), 8);
        assert_eq!(index.num_parents(), 2);
        assert_eq!(index.left_len(), 8);
        assert_eq!(index.right_len(), 8);
    }

    #[test]
    fn test_split_seqhash_new_odd_length() {
        let parents: Vec<&[u8]> = vec![b"ACGTACGTACGTACG", b"GGGGGCCCCGGGGCC"];

        let index = SplitSeqHash::new(&parents).unwrap();

        assert_eq!(index.seq_len(), 15);
        assert_eq!(index.split_pos(), 7);
        assert_eq!(index.left_len(), 7);
        assert_eq!(index.right_len(), 8);
    }

    #[test]
    fn test_split_seqhash_with_split_pos() {
        let parents: Vec<&[u8]> = vec![b"ACGTACGTACGTACGT"];

        let index = SplitSeqHash::with_split_pos(&parents, 6).unwrap();

        assert_eq!(index.seq_len(), 16);
        assert_eq!(index.split_pos(), 6);
        assert_eq!(index.left_len(), 6);
        assert_eq!(index.right_len(), 10);
    }

    #[test]
    fn test_split_seqhash_empty_parents() {
        let parents: Vec<&[u8]> = vec![];
        let result = SplitSeqHash::new(&parents);
        assert!(matches!(result, Err(SeqHashError::EmptyParents)));
    }

    #[test]
    fn test_split_seqhash_inconsistent_length() {
        let parents: Vec<&[u8]> = vec![b"ACGTACGT", b"GGGGCCCC", b"TTTTAA"];
        let result = SplitSeqHash::new(&parents);
        assert!(matches!(
            result,
            Err(SeqHashError::InconsistentLength { .. })
        ));
    }

    #[test]
    fn test_split_seqhash_invalid_split_pos_zero() {
        let parents: Vec<&[u8]> = vec![b"ACGTACGT"];
        let result = SplitSeqHash::with_split_pos(&parents, 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_split_seqhash_invalid_split_pos_too_large() {
        let parents: Vec<&[u8]> = vec![b"ACGTACGT"];
        let result = SplitSeqHash::with_split_pos(&parents, 8);
        assert!(result.is_err());
    }

    #[test]
    fn test_split_seqhash_query_exact_match() {
        let parents: Vec<&[u8]> = vec![b"ACGTACGTACGTACGT", b"GGGGCCCCGGGGCCCC"];

        let index = SplitSeqHash::new(&parents).unwrap();

        let result = index.query(b"ACGTACGTACGTACGT");
        assert_eq!(result.agreed_idx(), Some(0));
        assert_eq!(result.matched_hdist(), Some(0));
    }

    #[test]
    fn test_split_seqhash_query_mismatch_left() {
        let parents: Vec<&[u8]> = vec![b"ACGTACGTACGTACGT", b"GGGGCCCCGGGGCCCC"];

        let index = SplitSeqHash::new(&parents).unwrap();

        // One mismatch in left half
        let result = index.query(b"NCGTACGTACGTACGT");
        assert_eq!(result.agreed_idx(), Some(0));
        assert_eq!(result.matched_hdist(), Some(1));
    }

    #[test]
    fn test_split_seqhash_query_mismatch_right() {
        let parents: Vec<&[u8]> = vec![b"ACGTACGTACGTACGT", b"GGGGCCCCGGGGCCCC"];

        let index = SplitSeqHash::new(&parents).unwrap();

        // One mismatch in right half
        let result = index.query(b"ACGTACGTACGTACGN");
        assert_eq!(result.agreed_idx(), Some(0));
        assert_eq!(result.matched_hdist(), Some(1));
    }

    #[test]
    fn test_split_seqhash_query_mismatch_both() {
        let parents: Vec<&[u8]> = vec![b"ACGTACGTACGTACGT", b"GGGGCCCCGGGGCCCC"];

        let index = SplitSeqHash::new(&parents).unwrap();

        // One mismatch in each half
        let result = index.query(b"NCGTACGTACGTACGN");
        assert_eq!(result.agreed_idx(), Some(0));
        assert_eq!(result.matched_hdist(), Some(2));
    }

    #[test]
    fn test_split_seqhash_query_no_match() {
        let parents: Vec<&[u8]> = vec![b"ACGTACGTACGTACGT", b"GGGGCCCCGGGGCCCC"];

        let index = SplitSeqHash::new(&parents).unwrap();

        // Too many mismatches
        let result = index.query(b"NNNNNNNNNNNNNNNN");
        assert_eq!(result.agreed_idx(), None);
        assert_eq!(result.single_match(), None);
    }

    #[test]
    fn test_split_seqhash_query_single_match() {
        let parents: Vec<&[u8]> = vec![b"ACGTACGTACGTACGT", b"GGGGCCCCGGGGCCCC"];

        let index = SplitSeqHash::new(&parents).unwrap();

        // Only left half matches (right half has too many mismatches)
        let result = index.query(b"ACGTACGTNNNNNNNN");
        assert_eq!(result.agreed_idx(), None);
        assert_eq!(result.single_match(), Some((0, Half::Left)));
    }

    #[test]
    fn test_split_seqhash_query_conflict() {
        let parents: Vec<&[u8]> = vec![
            b"ACGTACGTACGTACGT",
            b"GGGGCCCCGGGGCCCC",
            b"TTTTAAAATTTTAAAA",
        ];

        let index = SplitSeqHash::new(&parents).unwrap();

        // Construct a query where left half matches parent 0 and right half matches parent 1
        // Left: ACGTACGT (matches parent 0)
        // Right: GGGGCCCC (matches parent 1)
        let result = index.query(b"ACGTACGTGGGGCCCC");
        assert!(result.is_conflicted());
        assert_eq!(result.agreed_idx(), None);
    }

    #[test]
    #[should_panic(expected = "Query sequence length")]
    fn test_split_seqhash_query_wrong_length() {
        let parents: Vec<&[u8]> = vec![b"ACGTACGTACGTACGT"];
        let index = SplitSeqHash::new(&parents).unwrap();
        let _ = index.query(b"ACGT");
    }

    #[test]
    fn test_split_seqhash_is_within_hdist_left() {
        let parents: Vec<&[u8]> = vec![b"ACGTACGTACGTACGT", b"GGGGCCCCGGGGCCCC"];

        let index = SplitSeqHash::new(&parents).unwrap();

        // Left half: ACGTACGT vs NCGTACGT (hdist=1)
        assert!(index.is_within_hdist(b"NCGTACGTXXXXXXXX", 0, Half::Left, 1));
        assert!(!index.is_within_hdist(b"NNGTACGTXXXXXXXX", 0, Half::Left, 1));
        assert!(index.is_within_hdist(b"NNGTACGTXXXXXXXX", 0, Half::Left, 2));
    }

    #[test]
    fn test_split_seqhash_is_within_hdist_right() {
        let parents: Vec<&[u8]> = vec![b"ACGTACGTACGTACGT", b"GGGGCCCCGGGGCCCC"];

        let index = SplitSeqHash::new(&parents).unwrap();

        // Right half: ACGTACGT vs ACGTACGN (hdist=1)
        assert!(index.is_within_hdist(b"XXXXXXXXACGTACGN", 0, Half::Right, 1));
        assert!(!index.is_within_hdist(b"XXXXXXXXACGTACNN", 0, Half::Right, 1));
        assert!(index.is_within_hdist(b"XXXXXXXXACGTACNN", 0, Half::Right, 2));
    }

    #[test]
    fn test_split_seqhash_is_within_hdist_invalid_length() {
        let parents: Vec<&[u8]> = vec![b"ACGTACGTACGTACGT"];
        let index = SplitSeqHash::new(&parents).unwrap();

        assert!(!index.is_within_hdist(b"ACGT", 0, Half::Left, 0));
    }

    #[test]
    fn test_split_seqhash_is_within_hdist_invalid_parent() {
        let parents: Vec<&[u8]> = vec![b"ACGTACGTACGTACGT"];
        let index = SplitSeqHash::new(&parents).unwrap();

        assert!(!index.is_within_hdist(b"ACGTACGTACGTACGT", 99, Half::Left, 0));
    }

    #[test]
    fn test_split_seqhash_fallback_pattern() {
        // This test replicates the cyto pattern (i.e. 3 mismatch tolerance with perfect match on one side)
        let parents: Vec<&[u8]> = vec![
            b"ACGTACGTACGTACGT",
            b"GGGGCCCCGGGGCCCC",
            b"TTTTAAAATTTTAAAA",
        ];

        let index = SplitSeqHash::new(&parents).unwrap();
        const MAX_HDIST: usize = 3;

        // Helper function mimicking the cyto pattern
        fn map_sequence(index: &SplitSeqHash, sequence: &[u8], max_hdist: usize) -> Option<usize> {
            let result = index.query(sequence);

            // Fast path: both halves agree
            if let Some(idx) = result.agreed_idx() {
                return Some(idx);
            }

            // Conflict: both matched but to different parents
            if result.is_conflicted() {
                return None;
            }

            // Fallback: one side matched, validate the other side with remaining budget
            if let Some((idx, matched_half)) = result.single_match() {
                let remaining = result.remaining_hdist(max_hdist).unwrap_or(0);

                if index.is_within_hdist(sequence, idx, matched_half.other(), remaining) {
                    return Some(idx);
                }
            }

            None
        }

        // Exact match - both halves agree
        assert_eq!(
            map_sequence(&index, b"ACGTACGTACGTACGT", MAX_HDIST),
            Some(0)
        );

        // One mismatch in left half - still agrees
        assert_eq!(
            map_sequence(&index, b"NCGTACGTACGTACGT", MAX_HDIST),
            Some(0)
        );

        // Two mismatches in left half - left half fails, but fallback validates
        assert_eq!(
            map_sequence(&index, b"NNGTACGTACGTACGT", MAX_HDIST),
            Some(0)
        );

        // Three mismatches total (2 left, 1 right) - within budget
        assert_eq!(
            map_sequence(&index, b"NNGTACGTACGTACGN", MAX_HDIST),
            Some(0)
        );

        // Four mismatches - exceeds budget
        assert_eq!(map_sequence(&index, b"NNGTACGTACGTACNN", MAX_HDIST), None);

        // Conflict case
        let conflict_query = b"ACGTACGTGGGGCCCC";
        assert_eq!(map_sequence(&index, conflict_query, MAX_HDIST), None);
    }

    #[test]
    fn test_split_seqhash_multiple_parents() {
        let parents: Vec<&[u8]> = vec![b"AAAAAAAA", b"CCCCCCCC", b"GGGGGGGG", b"TTTTTTTT"];

        let index = SplitSeqHash::new(&parents).unwrap();

        // Test each parent
        for i in 0..4 {
            let result = index.query(parents[i]);
            assert_eq!(result.agreed_idx(), Some(i));
        }
    }

    #[test]
    fn test_match_hdist() {
        let exact = Match::Exact { parent_idx: 0 };
        let mismatch = Match::Mismatch {
            parent_idx: 0,
            pos: 5,
        };

        assert_eq!(exact.hdist(), 0);
        assert_eq!(mismatch.hdist(), 1);
    }

    // ========================================================================
    // Positional query tests
    // ========================================================================

    #[test]
    fn test_split_match_has_match() {
        let both = SplitMatch {
            left: Some(Match::Exact { parent_idx: 0 }),
            right: Some(Match::Exact { parent_idx: 0 }),
        };
        assert!(both.has_match());

        let left_only = SplitMatch {
            left: Some(Match::Exact { parent_idx: 0 }),
            right: None,
        };
        assert!(left_only.has_match());

        let right_only = SplitMatch {
            left: None,
            right: Some(Match::Exact { parent_idx: 0 }),
        };
        assert!(right_only.has_match());

        let neither = SplitMatch {
            left: None,
            right: None,
        };
        assert!(!neither.has_match());
    }

    #[test]
    fn test_query_at_basic() {
        let parents: Vec<&[u8]> = vec![b"ACGTACGT", b"GGGGCCCC"];
        let index = SplitSeqHash::new(&parents).unwrap();

        // Target at position 2
        let read = b"NNACGTACGTNN";
        let result = index.query_at(read, 2);
        assert_eq!(result.agreed_idx(), Some(0));
    }

    #[test]
    fn test_query_at_with_mismatch() {
        let parents: Vec<&[u8]> = vec![b"ACGTACGT", b"GGGGCCCC"];
        let index = SplitSeqHash::new(&parents).unwrap();

        // Target at position 2 with one mismatch
        let read = b"NNNCGTACGTNN";
        let result = index.query_at(read, 2);
        assert_eq!(result.agreed_idx(), Some(0));
        assert_eq!(result.matched_hdist(), Some(1));
    }

    #[test]
    fn test_query_at_out_of_bounds() {
        let parents: Vec<&[u8]> = vec![b"ACGTACGT"];
        let index = SplitSeqHash::new(&parents).unwrap();

        let read = b"ACGT";

        // Position too far
        let result = index.query_at(read, 10);
        assert!(!result.has_match());

        // Would extend past end
        let result = index.query_at(read, 1);
        assert!(!result.has_match());
    }

    #[test]
    fn test_query_at_position_zero() {
        let parents: Vec<&[u8]> = vec![b"ACGTACGT"];
        let index = SplitSeqHash::new(&parents).unwrap();

        let read = b"ACGTACGTNNNN";
        let result = index.query_at(read, 0);
        assert_eq!(result.agreed_idx(), Some(0));
    }

    #[test]
    fn test_query_at_end_of_sequence() {
        let parents: Vec<&[u8]> = vec![b"ACGTACGT"];
        let index = SplitSeqHash::new(&parents).unwrap();

        let read = b"NNNNACGTACGT";
        let result = index.query_at(read, 4);
        assert_eq!(result.agreed_idx(), Some(0));
    }

    #[test]
    fn test_query_at_with_remap_exact_position() {
        let parents: Vec<&[u8]> = vec![b"ACGTACGT"];
        let index = SplitSeqHash::new(&parents).unwrap();

        let read = b"NNACGTACGTNN";
        let result = index.query_at_with_remap(read, 2, 3);
        assert_eq!(result.agreed_idx(), Some(0));
    }

    #[test]
    fn test_query_at_with_remap_positive_offset() {
        let parents: Vec<&[u8]> = vec![b"ACGTACGT"];
        let index = SplitSeqHash::new(&parents).unwrap();

        // Target is at position 4, but we look at position 2
        let read = b"NNNNACGTACGTNN";
        let result = index.query_at_with_remap(read, 2, 3);
        assert_eq!(result.agreed_idx(), Some(0));
    }

    #[test]
    fn test_query_at_with_remap_negative_offset() {
        let parents: Vec<&[u8]> = vec![b"ACGTACGT"];
        let index = SplitSeqHash::new(&parents).unwrap();

        // Target is at position 1, but we look at position 3
        // Read: NACGTACGTN (10 chars)
        // Position 1: ACGTACGT (8 chars) - exact match
        // Position 3: GTACGTNN - partial match (left half only)
        // The algorithm tries +1 first (pos 4: out of bounds for 8 chars in 10 char read)
        // Then tries -1 (pos 2: CGTACGTN - no match)
        // Then tries +2 (pos 5: out of bounds)
        // Then tries -2 (pos 1: ACGTACGT - full match!)
        let read = b"NACGTACGTN";
        let result = index.query_at_with_remap(read, 3, 3);
        assert_eq!(result.agreed_idx(), Some(0));
    }

    #[test]
    fn test_query_at_with_remap_offset_returns_correct_offset() {
        let parents: Vec<&[u8]> = vec![b"ACGTACGT"];
        let index = SplitSeqHash::new(&parents).unwrap();

        // Target at position 2, looking at position 2 -> offset 0
        let read = b"NNACGTACGTNN";
        let (result, offset) = index.query_at_with_remap_offset(read, 2, 3);
        assert_eq!(result.agreed_idx(), Some(0));
        assert_eq!(offset, 0);

        // Target at position 4, looking at position 2 -> offset +2
        let read = b"NNNNACGTACGTNN";
        let (result, offset) = index.query_at_with_remap_offset(read, 2, 3);
        assert_eq!(result.agreed_idx(), Some(0));
        assert_eq!(offset, 2);

        // Target at position 1, looking at position 4 -> offset -3
        // Read: NACGTACGTNNNN (13 chars)
        // Position 4: ACGTNNNN - partial match only (left half ACGT matches)
        // Position 5: CGTNNNN - no match (out of bounds anyway for 8 chars)
        // Position 3: GTACGTNN - partial match
        // Position 1: ACGTACGT - full match! offset -3
        // However, the algorithm finds partial match at pos 4 first (offset 0)
        // Let's use a read where no partial matches exist before the full match
        let read = b"NACGTACGTN";
        let (result, offset) = index.query_at_with_remap_offset(read, 4, 3);
        // Position 4: out of bounds (4+8=12 > 10)
        // Position 5: out of bounds
        // Position 3: GT... no match
        // Position 6: out of bounds
        // Position 2: CGTACGTN - no match
        // Position 7: out of bounds
        // Position 1: ACGTACGT - full match!
        assert_eq!(result.agreed_idx(), Some(0));
        assert_eq!(offset, -3);
    }

    #[test]
    fn test_query_at_with_remap_no_match_in_window() {
        let parents: Vec<&[u8]> = vec![b"ACGTACGT"];
        let index = SplitSeqHash::new(&parents).unwrap();

        // Target at position 10, but window only goes to +/- 3 from position 2
        let read = b"NNNNNNNNNNACGTACGT";
        let (result, offset) = index.query_at_with_remap_offset(read, 2, 3);
        assert!(!result.has_match());
        assert_eq!(offset, 0);
    }

    #[test]
    fn test_query_at_with_remap_partial_match() {
        let parents: Vec<&[u8]> = vec![b"ACGTACGTACGTACGT"];
        let index = SplitSeqHash::new(&parents).unwrap();

        // Only left half matches (right half has too many mismatches)
        let read = b"NNACGTACGTNNNNNNNNNN";
        let (result, offset) = index.query_at_with_remap_offset(read, 2, 3);

        // Should still find it because at least one half matched
        assert!(result.has_match());
        assert!(result.single_match().is_some());
        assert_eq!(offset, 0);
    }

    #[test]
    fn test_query_at_with_remap_prefers_earlier_offset() {
        let parents: Vec<&[u8]> = vec![b"ACGTACGT"];
        let index = SplitSeqHash::new(&parents).unwrap();

        // target only at position 3
        let read = b"NNNACGTACGTNN";
        let (result, offset) = index.query_at_with_remap_offset(read, 2, 3);
        assert_eq!(result.agreed_idx(), Some(0));
        assert_eq!(offset, 1); // +1 is tried before -1
    }

    #[test]
    fn test_query_sliding_basic() {
        let parents: Vec<&[u8]> = vec![b"ACGTACGT"];
        let index = SplitSeqHash::new(&parents).unwrap();

        let read = b"NNNACGTACGTNNN";
        let result = index.query_sliding(read);

        assert!(result.is_some());
        let (split_match, pos) = result.unwrap();
        assert_eq!(split_match.agreed_idx(), Some(0));
        assert_eq!(pos, 3);
    }

    #[test]
    fn test_query_sliding_at_start() {
        let parents: Vec<&[u8]> = vec![b"ACGTACGT"];
        let index = SplitSeqHash::new(&parents).unwrap();

        let read = b"ACGTACGTNNN";
        let result = index.query_sliding(read);

        assert!(result.is_some());
        let (split_match, pos) = result.unwrap();
        assert_eq!(split_match.agreed_idx(), Some(0));
        assert_eq!(pos, 0);
    }

    #[test]
    fn test_query_sliding_at_end() {
        let parents: Vec<&[u8]> = vec![b"ACGTACGT"];
        let index = SplitSeqHash::new(&parents).unwrap();

        let read = b"NNNACGTACGT";
        let result = index.query_sliding(read);

        assert!(result.is_some());
        let (split_match, pos) = result.unwrap();
        assert_eq!(split_match.agreed_idx(), Some(0));
        assert_eq!(pos, 3);
    }

    #[test]
    fn test_query_sliding_no_match() {
        let parents: Vec<&[u8]> = vec![b"ACGTACGT"];
        let index = SplitSeqHash::new(&parents).unwrap();

        let read = b"NNNNNNNNNNNN";
        let result = index.query_sliding(read);

        assert!(result.is_none());
    }

    #[test]
    fn test_query_sliding_sequence_too_short() {
        let parents: Vec<&[u8]> = vec![b"ACGTACGT"];
        let index = SplitSeqHash::new(&parents).unwrap();

        let read = b"ACGT"; // Shorter than seq_len
        let result = index.query_sliding(read);

        assert!(result.is_none());
    }

    #[test]
    fn test_query_sliding_iter_multiple_matches() {
        let parents: Vec<&[u8]> = vec![b"AAAATTTT"];
        let index = SplitSeqHash::new(&parents).unwrap();

        // Two occurrences of the target separated by X's
        let read = b"AAAATTTTXXXXXXXXAAAATTTT";
        let matches: Vec<_> = index.query_sliding_iter(read).collect();

        // The iterator finds all matches including partial ones (where only one half matches).
        // Due to SeqHash's 1-mismatch tolerance, positions adjacent to exact matches may
        // also produce partial matches. We verify:
        // 1. At least 2 matches are found (the exact matches at positions 0 and 16)
        // 2. The first and last matches are the exact matches we expect
        assert!(matches.len() >= 2);

        // First match should be at position 0 (exact match)
        assert_eq!(matches[0].1, 0);
        assert_eq!(matches[0].0.agreed_idx(), Some(0));

        // Last match should be at position 16 (exact match)
        let last = matches.last().unwrap();
        assert_eq!(last.1, 16);
        assert_eq!(last.0.agreed_idx(), Some(0));

        // Filter to only agreed (full) matches
        let full_matches: Vec<_> = matches
            .iter()
            .filter(|(m, _)| m.agreed_idx().is_some())
            .collect();
        assert_eq!(full_matches.len(), 2);
    }

    #[test]
    fn test_query_sliding_iter_overlapping_matches() {
        // This tests when matches could overlap
        let parents: Vec<&[u8]> = vec![b"AAAAAAAA"];
        let index = SplitSeqHash::new(&parents).unwrap();

        let read = b"AAAAAAAAAA"; // 10 A's, seq_len is 8
        let matches: Vec<_> = index.query_sliding_iter(read).collect();

        // Should find matches at positions 0, 1, 2
        assert_eq!(matches.len(), 3);
        assert_eq!(matches[0].1, 0);
        assert_eq!(matches[1].1, 1);
        assert_eq!(matches[2].1, 2);
    }

    #[test]
    fn test_query_sliding_iter_with_mismatches() {
        let parents: Vec<&[u8]> = vec![b"ACGTACGT"];
        let index = SplitSeqHash::new(&parents).unwrap();

        // Target with one mismatch
        let read = b"NNNNCGTACGTNN";
        let matches: Vec<_> = index.query_sliding_iter(read).collect();

        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].0.agreed_idx(), Some(0));
        assert_eq!(matches[0].0.matched_hdist(), Some(1));
        assert_eq!(matches[0].1, 3);
    }

    #[test]
    fn test_query_sliding_iter_lazy() {
        let parents: Vec<&[u8]> = vec![b"ACGTACGT"];
        let index = SplitSeqHash::new(&parents).unwrap();

        // Multiple matches, but we only take the first
        let read = b"ACGTACGTNNACGTACGTNNACGTACGT";
        let first_match = index.query_sliding_iter(read).next();

        assert!(first_match.is_some());
        let (split_match, pos) = first_match.unwrap();
        assert_eq!(split_match.agreed_idx(), Some(0));
        assert_eq!(pos, 0);
    }

    #[test]
    fn test_query_sliding_partial_match() {
        let parents: Vec<&[u8]> = vec![b"ACGTACGTACGTACGT"];
        let index = SplitSeqHash::new(&parents).unwrap();

        // Only left half will match
        let read = b"NNACGTACGTNNNNNNNNNN";
        let result = index.query_sliding(read);

        // Should find it because at least one half matched
        assert!(result.is_some());
        let (split_match, pos) = result.unwrap();
        assert!(split_match.has_match());
        assert!(split_match.single_match().is_some());
        assert_eq!(pos, 2);
    }

    #[test]
    fn test_query_at_with_remap_window_zero() {
        let parents: Vec<&[u8]> = vec![b"ACGTACGT"];
        let index = SplitSeqHash::new(&parents).unwrap();

        // With window=0, only exact position is checked
        let read = b"NNACGTACGTNN";

        // Exact position has the target
        let (result, offset) = index.query_at_with_remap_offset(read, 2, 0);
        assert_eq!(result.agreed_idx(), Some(0));
        assert_eq!(offset, 0);

        // Wrong position, window=0 means no remapping
        let (result, offset) = index.query_at_with_remap_offset(read, 3, 0);
        assert!(!result.has_match());
        assert_eq!(offset, 0);
    }

    #[test]
    fn test_query_methods_with_multiple_parents() {
        let parents: Vec<&[u8]> = vec![b"AAAACCCC", b"GGGGTTTT", b"ACGTACGT"];
        let index = SplitSeqHash::new(&parents).unwrap();

        // Test exact match at position 2
        let read = b"XXGGGGTTTTXX";

        // query_at - exact match at position 2
        let result = index.query_at(read, 2);
        assert_eq!(result.agreed_idx(), Some(1));

        // query_at_with_remap starting at position 2 with window 0
        // should find exact match at offset 0
        let result = index.query_at_with_remap(read, 2, 0);
        assert_eq!(result.agreed_idx(), Some(1));

        // query_sliding - finds exact match at position 2
        // (earlier positions may have partial matches depending on SeqHash indexing)
        let result = index.query_sliding(read);
        assert!(result.is_some());
        let (split_match, _pos) = result.unwrap();
        // The sliding search finds any match (partial or full), so we just verify
        // that it finds parent 1 somewhere (either as agreed or single match)
        let found_parent = split_match
            .agreed_idx()
            .or_else(|| split_match.single_match().map(|(idx, _)| idx));
        assert_eq!(found_parent, Some(1));
    }

    #[test]
    fn test_num_entries() {
        let parents: Vec<&[u8]> = vec![b"AAAACCCC", b"GGGGTTTT", b"ACGTACGT"];
        let index = SplitSeqHash::new(&parents).unwrap();
        assert_eq!(
            index.num_entries(),
            index.left.num_entries() + index.right.num_entries(),
        )
    }
}

#[cfg(all(test, feature = "serde"))]
mod serde_tests {
    use super::*;

    #[test]
    fn test_split_seqhash_serde_json() {
        let parents: Vec<&[u8]> = vec![
            b"ACGTACGTACGTACGT",
            b"GGGGCCCCGGGGCCCC",
            b"TTTTAAAATTTTAAAA",
        ];

        let index = SplitSeqHash::new(&parents).unwrap();

        // Serialize to JSON
        let json = serde_json::to_string(&index).unwrap();

        // Deserialize from JSON
        let restored: SplitSeqHash = serde_json::from_str(&json).unwrap();

        // Verify structure
        assert_eq!(restored.seq_len(), index.seq_len());
        assert_eq!(restored.split_pos(), index.split_pos());
        assert_eq!(restored.num_parents(), index.num_parents());

        // Verify queries work the same
        for parent in &parents {
            let original_result = index.query(parent);
            let restored_result = restored.query(parent);
            assert_eq!(original_result.agreed_idx(), restored_result.agreed_idx());
        }
    }

    #[test]
    fn test_split_seqhash_serde_bincode() {
        let parents: Vec<&[u8]> = vec![b"ACGTACGT", b"GGGGCCCC"];

        let index = SplitSeqHash::new(&parents).unwrap();

        // Serialize with bincode
        let encoded = bincode::encode_to_vec(&index, bincode::config::standard()).unwrap();

        // Deserialize with bincode
        let (restored, _): (SplitSeqHash, usize) =
            bincode::decode_from_slice(&encoded, bincode::config::standard()).unwrap();

        // Verify structure
        assert_eq!(restored.seq_len(), index.seq_len());
        assert_eq!(restored.split_pos(), index.split_pos());
        assert_eq!(restored.num_parents(), index.num_parents());

        // Verify functionality
        let result = restored.query(b"ACGTACGT");
        assert_eq!(result.agreed_idx(), Some(0));
    }

    #[test]
    fn test_split_match_serde() {
        let split_match = SplitMatch {
            left: Some(Match::Exact { parent_idx: 0 }),
            right: Some(Match::Mismatch {
                parent_idx: 0,
                pos: 5,
            }),
        };

        // Serialize to JSON
        let json = serde_json::to_string(&split_match).unwrap();

        // Deserialize from JSON
        let restored: SplitMatch = serde_json::from_str(&json).unwrap();

        // Verify fields
        assert_eq!(restored.agreed_idx(), split_match.agreed_idx());
        assert_eq!(restored.matched_hdist(), split_match.matched_hdist());
        assert_eq!(restored.single_match(), split_match.single_match());
    }

    #[test]
    fn test_half_serde() {
        let left = Half::Left;
        let right = Half::Right;

        // Serialize
        let left_json = serde_json::to_string(&left).unwrap();
        let right_json = serde_json::to_string(&right).unwrap();

        // Deserialize
        let left_restored: Half = serde_json::from_str(&left_json).unwrap();
        let right_restored: Half = serde_json::from_str(&right_json).unwrap();

        assert_eq!(left_restored, Half::Left);
        assert_eq!(right_restored, Half::Right);
    }
}
