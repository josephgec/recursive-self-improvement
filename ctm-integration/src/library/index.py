"""Simple keyword-based rule index for retrieval."""

from __future__ import annotations

import re
from collections import Counter
from typing import Dict, List, Optional, Set, Tuple

from src.library.rule import VerifiedRule
from src.library.store import RuleStore


class RuleIndex:
    """Keyword-based index for rule retrieval.

    Uses TF-IDF-like scoring on rule descriptions, tags, and domains
    for retrieval without external dependencies.
    """

    def __init__(self, store: Optional[RuleStore] = None) -> None:
        """Initialize the index.

        Args:
            store: RuleStore to index. If None, creates an empty index.
        """
        self.store = store
        self._index: Dict[str, List[Tuple[str, float]]] = {}  # token -> [(rule_id, score)]
        self._rules: Dict[str, VerifiedRule] = {}

        if store:
            self.build_index()

    def build_index(self) -> None:
        """Build the keyword index from the store's rules."""
        self._index = {}
        self._rules = {}

        if not self.store:
            return

        rules = self.store.list_all()
        # Document frequency for IDF
        doc_freq: Counter = Counter()
        rule_tokens: Dict[str, List[str]] = {}

        for rule in rules:
            self._rules[rule.rule_id] = rule
            tokens = self._tokenize(rule)
            rule_tokens[rule.rule_id] = tokens
            unique_tokens = set(tokens)
            for token in unique_tokens:
                doc_freq[token] += 1

        total_docs = len(rules) if rules else 1

        for rule in rules:
            tokens = rule_tokens[rule.rule_id]
            token_counts = Counter(tokens)
            total_tokens = len(tokens) if tokens else 1

            for token, count in token_counts.items():
                tf = count / total_tokens
                idf = 1.0 + (total_docs / (1 + doc_freq.get(token, 0)))
                score = tf * idf

                if token not in self._index:
                    self._index[token] = []
                self._index[token].append((rule.rule_id, score))

    def retrieve(self, query: str, k: int = 5) -> List[VerifiedRule]:
        """Retrieve the top-k most relevant rules for a query.

        Args:
            query: Search query string.
            k: Number of results to return.

        Returns:
            List of VerifiedRule objects, sorted by relevance.
        """
        query_tokens = self._normalize_tokens(query.lower().split())

        # Score each rule
        rule_scores: Counter = Counter()
        for token in query_tokens:
            if token in self._index:
                for rule_id, score in self._index[token]:
                    rule_scores[rule_id] += score

        # Sort by score (descending) and return top-k
        top_ids = [rule_id for rule_id, _ in rule_scores.most_common(k)]
        return [self._rules[rid] for rid in top_ids if rid in self._rules]

    def _tokenize(self, rule: VerifiedRule) -> List[str]:
        """Extract tokens from a rule for indexing."""
        text_parts = [
            rule.description,
            rule.domain,
            " ".join(rule.tags),
            rule.source_code,
        ]
        text = " ".join(text_parts).lower()
        # Split on non-alphanumeric characters
        tokens = re.findall(r"[a-z0-9_]+", text)
        return self._normalize_tokens(tokens)

    def _normalize_tokens(self, tokens: List[str]) -> List[str]:
        """Normalize and filter tokens."""
        # Remove very short tokens and common Python keywords
        stop_words = {
            "def", "return", "if", "else", "for", "in", "and", "or", "not",
            "is", "the", "a", "an", "to", "of", "with", "from", "import",
            "as", "class", "self", "none", "true", "false",
        }
        return [
            t for t in tokens
            if len(t) > 1 and t not in stop_words
        ]
