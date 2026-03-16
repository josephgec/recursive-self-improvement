"""Structural drift signal using regex-based text analysis."""

import math
import re
from collections import Counter
from typing import Dict, List

from .base import DriftSignal, SignalResult


def _split_sentences(text: str) -> List[str]:
    """Split text into sentences using regex."""
    sentences = re.split(r'[.!?]+\s*', text.strip())
    return [s.strip() for s in sentences if s.strip()]


def _sentence_lengths(texts: List[str]) -> List[int]:
    """Get word counts for each sentence across all texts."""
    lengths = []
    for text in texts:
        for sent in _split_sentences(text):
            words = re.findall(r'\b\w+\b', sent)
            lengths.append(len(words))
    return lengths


def _clause_depth(sentence: str) -> int:
    """Estimate sentence nesting depth via clause counting.

    Counts subordinating conjunctions, relative pronouns, and nesting markers.
    """
    depth = 1
    subordinators = [
        r'\b(because|although|while|when|if|unless|since|after|before|'
        r'that|which|who|whom|whose|where|whereas|whenever|wherever|'
        r'whether|though|even\s+if|so\s+that|in\s+order\s+that)\b'
    ]
    for pattern in subordinators:
        matches = re.findall(pattern, sentence.lower())
        depth += len(matches)

    # Count parenthetical nesting
    depth += sentence.count('(')
    depth += sentence.count('[')

    return depth


def _depth_distribution(texts: List[str]) -> List[int]:
    """Get clause depth distribution for all sentences."""
    depths = []
    for text in texts:
        for sent in _split_sentences(text):
            depths.append(_clause_depth(sent))
    return depths


def _categorize_token(token: str) -> str:
    """Simple POS-tag-like categorization via heuristics."""
    token_lower = token.lower()

    # Determiners
    if token_lower in {'a', 'an', 'the', 'this', 'that', 'these', 'those',
                        'my', 'your', 'his', 'her', 'its', 'our', 'their'}:
        return 'DET'

    # Prepositions
    if token_lower in {'in', 'on', 'at', 'to', 'for', 'with', 'by', 'from',
                        'of', 'about', 'into', 'through', 'during', 'before',
                        'after', 'above', 'below', 'between', 'under', 'over'}:
        return 'PREP'

    # Conjunctions
    if token_lower in {'and', 'or', 'but', 'nor', 'yet', 'so', 'because',
                        'although', 'while', 'if', 'when', 'unless', 'since'}:
        return 'CONJ'

    # Pronouns
    if token_lower in {'i', 'me', 'you', 'he', 'him', 'she', 'her', 'it',
                        'we', 'us', 'they', 'them', 'who', 'whom', 'what',
                        'which', 'that', 'myself', 'yourself', 'itself'}:
        return 'PRON'

    # Auxiliaries / modals
    if token_lower in {'is', 'am', 'are', 'was', 'were', 'be', 'been',
                        'being', 'have', 'has', 'had', 'do', 'does', 'did',
                        'will', 'would', 'shall', 'should', 'may', 'might',
                        'can', 'could', 'must'}:
        return 'AUX'

    # Numbers
    if re.match(r'^\d+$', token):
        return 'NUM'

    # Adverbs (common suffixes)
    if token_lower.endswith('ly') and len(token_lower) > 3:
        return 'ADV'

    # Adjectives (common suffixes)
    if any(token_lower.endswith(suf) for suf in
           ['ful', 'less', 'ous', 'ive', 'able', 'ible', 'al', 'ical']):
        return 'ADJ'

    # Verbs (common suffixes)
    if any(token_lower.endswith(suf) for suf in
           ['ing', 'ed', 'ize', 'ise', 'ate', 'ify']):
        return 'VERB'

    # Nouns (common suffixes)
    if any(token_lower.endswith(suf) for suf in
           ['tion', 'sion', 'ment', 'ness', 'ity', 'ence', 'ance', 'er',
            'or', 'ist', 'ism']):
        return 'NOUN'

    # Default: noun
    return 'NOUN'


def _node_type_distribution(texts: List[str]) -> Dict[str, float]:
    """Get POS category distribution across texts."""
    counter: Counter = Counter()
    for text in texts:
        tokens = re.findall(r'\b\w+\b', text)
        for token in tokens:
            counter[_categorize_token(token)] += 1

    total = sum(counter.values())
    if total == 0:
        return {}
    return {k: v / total for k, v in counter.items()}


def _distribution_shift(
    dist_a: List[int], dist_b: List[int]
) -> float:
    """Compute distributional shift between two integer distributions.

    Uses normalized histogram comparison via chi-squared-like metric.
    """
    if not dist_a or not dist_b:
        return 1.0 if (dist_a or dist_b) else 0.0

    # Create histograms
    all_vals = dist_a + dist_b
    min_val = min(all_vals)
    max_val = max(all_vals)

    if min_val == max_val:
        return 0.0

    n_bins = min(20, max_val - min_val + 1)
    bin_width = (max_val - min_val + 1) / n_bins

    def _histogram(values: List[int]) -> List[float]:
        bins = [0.0] * n_bins
        for v in values:
            idx = min(int((v - min_val) / bin_width), n_bins - 1)
            bins[idx] += 1
        total = sum(bins)
        if total > 0:
            bins = [b / total for b in bins]
        return bins

    hist_a = _histogram(dist_a)
    hist_b = _histogram(dist_b)

    # Jensen-Shannon divergence of histograms
    divergence = 0.0
    for a, b in zip(hist_a, hist_b):
        m = (a + b) / 2
        if a > 0 and m > 0:
            divergence += 0.5 * a * math.log2(a / m)
        if b > 0 and m > 0:
            divergence += 0.5 * b * math.log2(b / m)

    return min(1.0, max(0.0, divergence))


def _dict_distribution_shift(
    dist_a: Dict[str, float], dist_b: Dict[str, float]
) -> float:
    """Compute JS divergence between two probability distributions."""
    if not dist_a and not dist_b:
        return 0.0
    if not dist_a or not dist_b:
        return 1.0

    all_keys = set(dist_a.keys()) | set(dist_b.keys())

    divergence = 0.0
    for k in all_keys:
        a = dist_a.get(k, 0.0)
        b = dist_b.get(k, 0.0)
        m = (a + b) / 2
        if a > 0 and m > 0:
            divergence += 0.5 * a * math.log2(a / m)
        if b > 0 and m > 0:
            divergence += 0.5 * b * math.log2(b / m)

    return min(1.0, max(0.0, divergence))


class StructuralDriftSignal(DriftSignal):
    """Structural drift signal using regex-based text analysis.

    Measures structural changes in sentence length, nesting depth,
    and syntactic category distributions without requiring spacy.
    """

    @property
    def name(self) -> str:
        return "structural"

    def sentence_length_distribution_shift(
        self, current: List[str], reference: List[str]
    ) -> float:
        """Shift in sentence length distributions."""
        cur_lengths = _sentence_lengths(current)
        ref_lengths = _sentence_lengths(reference)
        return _distribution_shift(cur_lengths, ref_lengths)

    def depth_distribution_shift(
        self, current: List[str], reference: List[str]
    ) -> float:
        """Shift in clause nesting depth distributions."""
        cur_depths = _depth_distribution(current)
        ref_depths = _depth_distribution(reference)
        return _distribution_shift(cur_depths, ref_depths)

    def node_type_shift(
        self, current: List[str], reference: List[str]
    ) -> float:
        """Shift in POS-tag-like category distributions."""
        cur_dist = _node_type_distribution(current)
        ref_dist = _node_type_distribution(reference)
        return _dict_distribution_shift(cur_dist, ref_dist)

    def compute(
        self, current: List[str], reference: List[str]
    ) -> SignalResult:
        """Compute structural drift as average of three components."""
        sent_shift = self.sentence_length_distribution_shift(current, reference)
        depth_shift = self.depth_distribution_shift(current, reference)
        type_shift = self.node_type_shift(current, reference)

        raw = (sent_shift + depth_shift + type_shift) / 3.0
        normalized = self.normalize(raw)

        return SignalResult(
            signal_name=self.name,
            raw_score=raw,
            normalized_score=normalized,
            interpretation=self.interpret(normalized),
            components={
                "sentence_length_shift": sent_shift,
                "depth_distribution_shift": depth_shift,
                "node_type_shift": type_shift,
            },
        )

    def normalize(self, raw: float) -> float:
        """Normalize: raw / 0.5, capped at 1.0."""
        return min(1.0, raw / 0.5)
